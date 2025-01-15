# Graph Representation Learning - Mini Project

This repository contains the code for the **Graph Representation Learning** mini-project, which focuses on studying **over-squashing** and designing two methods to mitigate it.

## Project Structure

The project is organized as follows:

### Jupyter Notebooks:
1. **`synthetic_data.ipynb`**: Contains code and results for the synthetic dataset.
2. **`Real_data.ipynb`**: Contains code and results for real-world datasets:
   - **PROTEINS** from TUData
   - **Peptides-functional** dataset

Both notebooks are designed to be run in **Google Colab**.

### Required Files:
- **`synthetic_data.ipynb`** requires the following Python scripts:
  - `data_generation.py`: Contains functions for generating synthetic data.
  - `synthetic_model.py`: Contains the model definition for the synthetic dataset.
- **`Real_data.ipynb`** requires the following dataset:
  - `peptide_multi_class_dataset.csv`: Contains the real-world peptide dataset.

All the above files can be downloaded from this repository.

### Result Folders:
1. **`Tree Visualization`**: Contains the visualization results of the tree experiments. Each file is named after the corresponding experiment.
2. **`Clustered Data Visualization`**: Similar to the previous folder, but these visualizations correspond to clustered graphs, as described in the notebook.

## How to Run

You can run both notebooks directly in **Google Colab**. Ensure the required files and datasets are available as mentioned above. For the synthetic data experiment, make sure both `data_generation.py` and `synthetic_model.py` are in the same directory as the notebook.

