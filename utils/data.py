from preprocessing.Preprocess import Preprocessor
from representations.sequences.statistics import Sequential_TF

import torch
from CONSTANTS import DEVICE, PROJECT_ROOT

from module.Common import data_iter, generate_tinsts_binary_label

import os
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def preprocess_data(dataset, parser, cut_func, template_encoder, target_mode=False, support_set=None):
    """
    Load and parse log data, segment into train/val/test sets, and encode templates.

    Args:
        dataset (str): Dataset name (e.g., "HDFS", "BGL").
        parser (str): Parsing method (e.g., "IBM" → Drain parser).
        cut_func (callable): Data splitting strategy (e.g., cut_by).
        template_encoder (object): Encoder with `.present()` method for embedding templates.
        target_mode (bool): If True, only process normal logs for target system.
        support_set (list[Instance], optional): If provided, only use these instances for building vocabulary.

    Returns:
        tuple: (train_data, valid_data, test_data, processor)
    """
    if not os.path.exists(os.path.join(PROJECT_ROOT, f"datasets/{dataset}")):
        raise ValueError(f"Dataset {dataset} not found in {PROJECT_ROOT}/datasets/")

    processor = Preprocessor()
    try:
        train_data, valid_data, test_data = processor.process(
            dataset=dataset,
            parsing=parser,
            cut_func=cut_func,
            template_encoding=template_encoder.present,
            target_mode=target_mode
        )
        
        # If support_set is provided, only keep embeddings for templates in support set
        if support_set is not None:
            support_templates = set()
            for inst in support_set:
                support_templates.update(inst.sequence)
            
            # Filter embeddings to only include templates from support set
            processor.embedding = {k: v for k, v in processor.embedding.items() 
                                if k in support_templates}
            
            # Update templates dictionary
            processor.templates = {k: v for k, v in processor.templates.items() 
                                if k in support_templates}
            
            logger.info(f"Filtered embeddings to {len(processor.embedding)} templates from support set")
            
        return train_data, valid_data, test_data, processor
    except Exception as e:
        logger.error(f"Error preprocessing {dataset}: {str(e)}")
        raise


def encode_log_sequences(processor, train_data, test_data=None):
    """
    Encode log sequences using template-based sequential TF encoder.

    Args:
        processor (Preprocessor): Contains template embeddings.
        train_data (list[Instance]): Training instances.
        test_data (list[Instance], optional): Optional test set.

    Returns:
        tuple: Updated (train_data, test_data) with `.repr` as semantic vector.
    """
    sequential_encoder = Sequential_TF(processor.embedding)

    train_reprs = sequential_encoder.present(train_data)
    for i, inst in enumerate(train_data):
        inst.repr = train_reprs[i]

    if test_data is not None:
        test_reprs = sequential_encoder.present(test_data)
        for i, inst in enumerate(test_data):
            inst.repr = test_reprs[i]
        return train_data, test_data

    return train_data, None


def encode_log_sequences_with_gru(model, vocab, instances, batch_size=128, show_progress=True):
    """
    Encode log sequences into latent vectors using AttGRUModel.

    Args:
        model (AttGRUModel): Encoder with attention GRU.
        vocab (Vocab): Vocabulary used for token indexing.
        instances (list[Instance]): List of log instances.
        batch_size (int): Batch size for processing.
        show_progress (bool): Whether to show progress bar.

    Returns:
        list[Instance]: Same list with `.repr` updated from latent space.
    """
    if not instances:
        raise ValueError("Empty instance list provided")

    model.eval()
    encoded_instances = []
    
    with torch.no_grad():
        iterator = data_iter(instances, batch_size=batch_size, shuffle=False)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding sequences", total=len(instances)//batch_size + 1)
            
        for batch in iterator:
            try:
                tinst = generate_tinsts_binary_label(batch, vocab)
                tinst.to(DEVICE)

                _, _, latent = model(tinst.inputs)
                for i, inst in enumerate(batch):
                    inst.repr = latent[i].detach().cpu().numpy()
                    encoded_instances.append(inst)
            except Exception as e:
                logger.error(f"Error encoding batch: {str(e)}")
                raise

    return encoded_instances


def find_most_similar_template(instance, source_encoders, similarity_threshold=0.8):
    """
    Find the most similar template from source systems based on sequence similarity.

    Args:
        instance (Instance): Target instance to find similar template for.
        source_encoders (dict[str, AttGRUModel]): Source encoders with repr_lookup.
        similarity_threshold (float): Minimum similarity threshold.

    Returns:
        np.ndarray or None: Best matching representation if found, None otherwise.
    """
    best_similarity = 0
    best_repr = None
    
    for system, encoder in source_encoders.items():
        if not hasattr(encoder, "repr_lookup"):
            continue
            
        for seq_key, repr in encoder.repr_lookup.items():
            # Calculate sequence similarity (simple Jaccard similarity for now)
            target_set = set(instance.sequence)
            source_set = set(seq_key)
            similarity = len(target_set.intersection(source_set)) / len(target_set.union(source_set))
            
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_repr = repr
                
    return best_repr


def fallback_encode_instance(instance, encoder_target, vocab_target, source_encoders, similarity_threshold=0.8):
    """
    Encode an instance using the target encoder. If template tokens are unseen in target vocab,
    fallback to a source encoder that has seen the same sequence (template id sequence).

    Args:
        instance (Instance): Log instance to encode.
        encoder_target (AttGRUModel): Encoder for the target system.
        vocab_target (Vocab): Target vocab (may not cover all tokens in query set).
        source_encoders (dict[str, AttGRUModel]): Source encoders with repr_lookup.
        similarity_threshold (float): Threshold for template similarity.

    Returns:
        np.ndarray: Latent representation vector.
    """
    seq_key = tuple(instance.sequence)

    try:
        # Check if sequence is known to target vocab
        _ = [vocab_target.word2id(token) for token in instance.sequence]

        # Encode using target encoder
        encoder_target.eval()
        tinst = generate_tinsts_binary_label([instance], vocab_target)
        tinst.to(DEVICE)
        _, _, latent = encoder_target(tinst.inputs)
        return latent[0].detach().cpu().numpy()

    except KeyError:
        # Token in sequence not found in target vocab → fallback
        # First try exact match
        for system, encoder in source_encoders.items():
            if hasattr(encoder, "repr_lookup") and seq_key in encoder.repr_lookup:
                return encoder.repr_lookup[seq_key]
                
        # If no exact match, try finding similar template
        best_match = find_most_similar_template(instance, source_encoders, similarity_threshold)
        if best_match is not None:
            return best_match
            
        raise ValueError(f"Instance {instance.id} cannot be encoded — no suitable fallback found in any source system.")


def encode_query_with_fallback(query_set, encoder_target, vocab_target, source_encoders, similarity_threshold=0.8):
    """
    Encode the query set of target system, using fallback mechanism for unseen templates.

    Args:
        query_set (list[Instance]): Query instances from target system.
        encoder_target (AttGRUModel): Target encoder.
        vocab_target (Vocab): Target vocab.
        source_encoders (dict[str, AttGRUModel]): Source encoders with fallback.
        similarity_threshold (float): Threshold for template similarity.

    Returns:
        list[Instance]: Query instances with `.repr` assigned.
    """
    if not query_set:
        raise ValueError("Empty query set provided")

    encoded = []
    fallback_count = 0
    total_count = len(query_set)
    
    for inst in tqdm(query_set, desc="Encoding query set"):
        try:
            inst.repr = fallback_encode_instance(
                inst, 
                encoder_target, 
                vocab_target, 
                source_encoders,
                similarity_threshold
            )
            encoded.append(inst)
        except ValueError as e:
            fallback_count += 1
            logger.warning(f"Fallback for instance {inst.id}: {str(e)}")
            # Try to find most similar template from source systems
            best_match = find_most_similar_template(inst, source_encoders, similarity_threshold)
            if best_match:
                inst.repr = best_match
                encoded.append(inst)
            else:
                logger.error(f"No suitable fallback found for instance {inst.id}")

    logger.info(f"Fallback statistics: {fallback_count}/{total_count} instances required fallback")
    return encoded


