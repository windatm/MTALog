from preprocessing.Preprocess import Preprocessor
from representations.sequences.statistics import Sequential_TF
import torch.nn.functional as F 
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
    if not instances:
        logger.warning("Empty instance list provided to encode_log_sequences_with_gru. Returning empty list.")
        return []

    # Ensure model is available
    if model is None:
        raise ValueError("No model provided for encoding")
    
    # Number of successful encodings
    success_count = 0
    
    # Make sure model is on the correct device
    model = model.to(DEVICE)    
    model.eval()
    encoded_instances = []

    logger.info(f"Starting encoding of {len(instances)} instances with batch size {batch_size}")

    with torch.no_grad():
        logger.info(f"Using batch size: {batch_size} for more stable processing")
        
        iterator = data_iter(instances, batch_size=batch_size, shuffle=False)
        
        # Skip iterator creation if no instances (defensive)
        total = max(1, len(instances)//batch_size) if instances else 0
        if show_progress and total > 0:
            iterator = tqdm(iterator, desc="Encoding sequences", total=total)

        for batch_idx, batch in enumerate(iterator):
            if not batch:  # Skip empty batches
                continue
                
            try:
                logger.info(f"Processing batch {batch_idx+1}/{total}, size: {len(batch)}")
                
                # Generate tensor instances from batch
                tinst_result = generate_tinsts_binary_label(batch, vocab)
                
                # Verify that tinst_result is not None and has the expected format
                if tinst_result is None or len(tinst_result) != 2:
                    logger.error(f"Invalid result from generate_tinsts_binary_label: {tinst_result}")
                    continue
                    
                # Extract tinst and validate it
                tinst, _ = tinst_result
                
                # Check if tinst is None or doesn't have the required attribute
                if tinst is None:
                    logger.error("generate_tinsts_binary_label returned None for tinst")
                    continue
                
                # Check if tinst has the inputs attribute
                if not hasattr(tinst, 'inputs'):
                    logger.error(f"tinst object does not have 'inputs' attribute. Type: {type(tinst)}")
                    continue
                
                # Verify inputs exists and is correctly formed
                inputs = getattr(tinst, 'inputs', None)
                if inputs is None or not isinstance(inputs, tuple) or len(inputs) < 3:
                    logger.error(f"tinst.inputs is not properly formed: {inputs}")
                    continue
                
                # Make sure all tensors are on the same device
                # Note: tinst.to() doesn't modify in-place, need to assign back
                try:
                    tinst = tinst.to(DEVICE)
                    
                    # Explicitly move inputs to device
                    words, masks, word_len = tinst.inputs
                    words = words.to(DEVICE)
                    masks = masks.to(DEVICE)
                    word_len = word_len.to(DEVICE)
                    tinst.inputs = (words, masks, word_len)
                    
                    logger.info(f"Input shapes - Words: {words.shape}, Masks: {masks.shape}, Word_len: {word_len.shape}")
                except Exception as e:
                    logger.error(f"Error moving tensors to device: {str(e)}")
                    continue

                # Process through model
                try:
                    model_device = next(model.parameters()).device
                    logger.info(f"Model is on device: {model_device}, Inputs are on device: {words.device}")
                    
                    # Forward pass
                    results = model(tinst.inputs)
                    
                    if results is None or len(results) < 3:
                        logger.error(f"Model returned invalid result: {results}")
                        continue
                        
                    _, _, latent = results
                    
                    if latent is None:
                        logger.error("Model returned None for latent representation")
                        continue
                    
                    logger.info(f"Latent shape: {latent.shape}")
                    
                    # Normalize latent vectors
                    latent = F.normalize(latent, p=2, dim=1)

                    # Update instance representations
                    for i, inst in enumerate(batch):
                        inst.repr = latent[i].detach().cpu().numpy()
                        encoded_instances.append(inst)
                        success_count += 1
                        
                except Exception as e:
                    logger.error(f"Error in model processing: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue

            except Exception as e:
                logger.error(f"Error encoding batch: {str(e)}")
                # Print more detailed error diagnostics
                logger.error(f"Batch type: {type(batch)}, Length: {len(batch) if batch else 0}")
                if batch and len(batch) > 0:
                    logger.error(f"First instance type: {type(batch[0])}")
                    if hasattr(batch[0], 'sequence'):
                        logger.error(f"Sequence length: {len(batch[0].sequence)}")
                # Continue with the next batch instead of stopping
                import traceback
                logger.error(traceback.format_exc())
                continue

    logger.info(f"Encoding complete: {success_count}/{len(instances)} instances successfully encoded")
    return encoded_instances


def find_most_similar_template(instance, source_encoders, similarity_threshold=0.8):
    """
    Find the most similar template from source systems based on sequence similarity.

    Args:
        instance (Instance): Target instance to find similar template for.
        source_encoders (dict[str, AttGRUModel]): Source encoders with repr_lookup.
        similarity_threshold (float): Minimum similarity threshold between 0 and 1.

    Returns:
        np.ndarray or None: Best matching representation if found, None otherwise.
    """
    # Validate inputs
    if instance is None:
        logger.warning("None instance provided to find_most_similar_template. Returning None.")
        return None
        
    if not source_encoders:
        logger.warning("No source encoders provided to find_most_similar_template. Returning None.")
        return None
    
    # Validate similarity threshold
    if not (0 <= similarity_threshold <= 1):
        logger.warning(f"Invalid similarity threshold {similarity_threshold}. Using default 0.8.")
        similarity_threshold = 0.8
    
    # Check if instance has sequence attribute
    if not hasattr(instance, 'sequence') or not instance.sequence:
        logger.warning("Instance has no sequence or empty sequence. Returning None.")
        return None
    
    try:
        best_similarity = 0
        best_repr = None
        
        for system, encoder in source_encoders.items():
            if encoder is None:
                logger.warning(f"Encoder for system {system} is None. Skipping.")
                continue
                
            if not hasattr(encoder, "repr_lookup") or not encoder.repr_lookup:
                logger.warning(f"No repr_lookup found for encoder {system}. Skipping.")
                continue
                
            for seq_key, repr in encoder.repr_lookup.items():
                # Skip empty sequences
                if not seq_key:
                    continue
                    
                # Calculate sequence similarity (simple Jaccard similarity for now)
                try:
                    target_set = set(instance.sequence)
                    source_set = set(seq_key)
                        
                    # Avoid division by zero
                    union_size = len(target_set.union(source_set))
                    if union_size == 0:
                        continue
                        
                    similarity = len(target_set.intersection(source_set)) / union_size
                
                    if similarity > best_similarity and similarity >= similarity_threshold:
                        best_similarity = similarity
                        best_repr = repr
                except Exception as e:
                    logger.warning(f"Error calculating similarity: {str(e)}. Skipping this template.")
                    continue
        
        if best_repr is not None:
            logger.debug(f"Found similar template with similarity {best_similarity:.2f}")
                
        return best_repr
        
    except Exception as e:
        logger.error(f"Error in find_most_similar_template: {str(e)}")
        return None


def fallback_encode_instance(instance, model, source_vocab, target_vocab, similarity_threshold=0.6):
    """
    Encodes an instance even when some tokens are not found in target vocabulary.
    Uses a similarity-based fallback mechanism.
    """
    if instance is None:
        logger.warning("None instance provided to fallback_encode_instance. Returning None.")
        return None
        
    if model is None or source_vocab is None or target_vocab is None:
        logger.error("Missing required parameter in fallback_encode_instance")
        raise ValueError("Model, source_vocab, and target_vocab must all be provided")
    
    # First try encoding with the target vocabulary
    try:
        result = []
        for token in instance.template_ids:
            if token in target_vocab.w2i:
                result.append(target_vocab.w2i[token])
            else:
                # If token not in target vocab, find most similar template
                similar_template = find_most_similar_template(
                    token, model, source_vocab, target_vocab, similarity_threshold
                )
                
                if similar_template is not None:
                    result.append(target_vocab.w2i[similar_template])
                else:
                    # If no similar template found above threshold, use UNK
                    if "<UNK>" in target_vocab.w2i:
                        result.append(target_vocab.w2i["<UNK>"])
                    else:
                        # If no UNK token in vocab, use first token as fallback
                        logger.warning(f"No <UNK> token in vocabulary, using first token as fallback for: {token}")
                        result.append(next(iter(target_vocab.w2i.values())))
                        
        return result
    except Exception as e:
        logger.error(f"Error in fallback_encode_instance: {str(e)}")
        # Return empty list as last resort
        return []


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


