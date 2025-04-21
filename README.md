# MTALog: Meta-Transfer Learning for Log Anomaly Detection

MTALog is a Python implementation of a meta-transfer learning approach for log anomaly detection. The system allows for efficient few-shot learning on new log systems by transferring knowledge from source systems.

## Overview

Log anomaly detection is a critical task in system monitoring, but traditional methods require large amounts of labeled data for each new system. MTALog addresses this challenge by:

1. Using meta-learning to extract knowledge from source log systems
2. Transferring this knowledge to target systems with limited labeled data
3. Providing efficient few-shot learning capabilities

## Features

- **Meta-learning framework**: Learns across multiple log systems
- **GRU-based log sequence encoding**: Effective representation of log patterns
- **Few-shot learning**: Requires minimal labeled examples for new systems
- **Transfer learning**: Knowledge transfer between different log systems
- **Support for multiple log parsers**: Compatible with IBM, Drain, and Spell parsers
- **Command-line interface**: Easy to use for training, evaluation, and prediction

## Installation

### Prerequisites

- Python 3.6+
- PyTorch 1.7+

### Setup

1. Clone the repository:

```bash
git clone https://github.com/username/MTALog.git
cd MTALog
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download word embeddings (GloVe):

```bash
# For example, download GloVe embeddings
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

## Project Structure

```
MTALog/
├── models/             # Neural network model implementations
├── module/             # Core modules for the system
├── preprocessing/      # Log preprocessing and parsing
├── representations/    # Template representation methods
├── utils/              # Utility functions
├── conf/               # Configuration files for log parsers
├── datasets/           # Log datasets directory
├── logs/               # System logs directory
├── outputs/            # Output files directory
├── main.py             # Main implementation
├── run.py              # Command-line interface
├── training.py         # Training and evaluation functions
├── CONSTANTS.py        # System constants
└── requirements.txt    # Project dependencies
```

## Usage

### Training

To train the model on source systems and adapt to a target system:

```bash
python run.py --mode train \
    --source_systems HDFS OpenStack \
    --target_system BGL \
    --parser IBM \
    --epochs 5 \
    --batch_size 1024 \
    --few_shot_ratio 0.1
```

### Evaluation

To evaluate a trained model on a target system:

```bash
python run.py --mode eval \
    --target_system BGL \
    --model_path outputs/models/IBM/best_model_epoch_5.pt
```

### Prediction

To use a trained model for anomaly detection on new logs:

```bash
python run.py --mode predict \
    --target_system BGL \
    --model_path outputs/models/IBM/best_model_epoch_5.pt
```

### Command-line Arguments

| Argument               | Description                             | Default           |
| ---------------------- | --------------------------------------- | ----------------- |
| `--mode`               | Operation mode: train, eval, or predict | train             |
| `--source_systems`     | Source log systems for meta-learning    | HDFS OpenStack    |
| `--target_system`      | Target log system                       | BGL               |
| `--parser`             | Log parser to use                       | IBM               |
| `--hidden_size`        | Hidden size of the GRU encoder          | 64                |
| `--num_layers`         | Number of GRU layers                    | 4                 |
| `--dropout`            | Dropout rate                            | 0.5               |
| `--batch_size`         | Batch size for training                 | 1024              |
| `--epochs`             | Number of training epochs               | 5                 |
| `--alpha`              | Inner loop learning rate                | 0.008             |
| `--beta`               | Outer loop scaling factor               | 1.0               |
| `--gamma`              | Learning rate for optimizer             | 0.008             |
| `--few_shot_ratio`     | Ratio of normal logs in support set     | 0.1               |
| `--query_sample_ratio` | Ratio of query set sampled              | 1.0               |
| `--model_path`         | Path to saved model checkpoint          | None              |
| `--word2vec_file`      | Word2Vec embeddings file                | glove.6B.300d.txt |

## Dataset Preparation

MTALog expects log datasets to be in specific directories with specific formats:

1. Place raw log files in `datasets/{system_name}/` directory
2. Run preprocessing scripts to parse logs
3. The system will automatically handle the rest

### Supported Log Systems

- HDFS
- BGL
- OpenStack
- Thunderbird
- (Others can be added with appropriate preprocessors)

## Configuration

Adjust parser configurations in the `conf/` directory for different log systems:

```
conf/
├── BGL.ini       # BGL parser configuration
├── HDFS.ini      # HDFS parser configuration
└── OpenStack.ini # OpenStack parser configuration
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use MTALog in your research, please cite:

```
@article{mtalog2023,
  title={MTALog: A Meta-Transfer Learning Approach for Log Anomaly Detection},
  author={Your Name},
  journal={Journal Name},
  year={2023}
}
```

## Acknowledgements

- This project is inspired by the research in meta-learning and transfer learning for log analysis
  conda create -n mtalog_env python=3.11
  conda activate mtalog_env
  pip install -r requirements.txts
