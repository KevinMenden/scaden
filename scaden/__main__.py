import click
import scaden
import logging
import os
from scaden.train import training
from scaden.predict import prediction
from scaden.process import processing
from scaden.simulate import simulation
from scaden.example import exampleData
"""

author: Kevin Menden

This is the main file for executing the Scaden program.
"""

# Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def main():
    text = """
     ____                _            
    / ___|  ___ __ _  __| | ___ _ __  
    \___ \ / __/ _` |/ _` |/ _ \ '_ \ 
     ___) | (_| (_| | (_| |  __/ | | |
    |____/ \___\__,_|\__,_|\___|_| |_|

    """
    click.echo(click.style(text, fg='blue'))
    cli()


if __name__ == '__main__':
    main()
"""
Set up the command line client with different commands to execute
"""


@click.group()
@click.version_option(scaden.__version__)
def cli():
    pass


"""
Training mode
"""


@cli.command()
@click.argument('data_path',
                type=click.Path(exists=True),
                required=True,
                metavar='<training data>')
@click.option(
    '--train_datasets',
    default='',
    help=
    'Comma-separated list of datasets used for training. Uses all by default.')
@click.option('--model_dir', default="./", help='Path to store the model in')
@click.option('--batch_size',
              default=128,
              help='Batch size to use for training. Default: 128')
@click.option('--learning_rate',
              default=0.0001,
              help='Learning rate used for training. Default: 0.0001')
@click.option('--steps', default=5000, help='Number of training steps')
@click.option('--seed', default=0, help="Set random seed")
def train(data_path, train_datasets, model_dir, batch_size, learning_rate,
          steps, seed):
    """ Train a Scaden model """
    training(data_path=data_path,
             train_datasets=train_datasets,
             model_dir=model_dir,
             batch_size=batch_size,
             learning_rate=learning_rate,
             num_steps=steps,
             seed=seed)


"""
Prediction mode
"""


@cli.command()
@click.argument('data_path',
                type=click.Path(exists=True),
                required=True,
                metavar='<prediction data>')
@click.option('--model_dir', default="./", help='Path to trained model')
@click.option('--outname',
              default="scaden_predictions.txt",
              help='Name of predictions file.')
@click.option('--seed', default=0, help="Set random seed")
def predict(data_path, model_dir, outname, seed):
    """ Predict cell type composition using a trained Scaden model"""
    prediction(model_dir=model_dir,
               data_path=data_path,
               out_name=outname,
               seed=seed)


"""
Processing mode
"""


@cli.command()
@click.argument('data_path',
                type=click.Path(exists=True),
                required=True,
                metavar='<training dataset to be processed>')
@click.argument('prediction_data',
                type=click.Path(exists=True),
                required=True,
                metavar='<data for prediction>')
@click.option('--processed_path',
              default="processed.h5ad",
              help='Path of processed file. Must end with .h5ad')
@click.option(
    '--var_cutoff',
    default=0.1,
    help=
    'Filter out genes with a variance less than the specified cutoff. A low cutoff is recommended,'
    'this should only remove genes that are obviously uninformative.')
def process(data_path, prediction_data, processed_path, var_cutoff):
    """ Process a dataset for training """
    processing(data_path=prediction_data,
               training_data=data_path,
               processed_path=processed_path,
               var_cutoff=var_cutoff)


"""
Simulate dataset
"""


@cli.command()
@click.option('--out',
              '-o',
              default='./',
              help="Directory to store output files in")
@click.option('--data', '-d', default='.', help="Path to scRNA-seq dataset(s)")
@click.option('--cells',
              '-c',
              default=100,
              help="Number of cells per sample [default: 100]")
@click.option('--n_samples',
              '-n',
              default=1000,
              help="Number of samples to simulate [default: 1000]")
@click.option(
    '--pattern',
    default="*_counts.txt",
    help="File pattern to recognize your processed scRNA-seq count files")
@click.option(
    '--unknown',
    '-u',
    multiple=True,
    default=['unknown'],
    help=
    "Specifiy cell types to merge into the unknown category. Specify this flag for every cell type you want to merge in unknown. [default: unknown]"
)
@click.option('--prefix',
              '-p',
              default="data",
              help="Prefix to append to training .h5ad file [default: data]")
def simulate(out, data, cells, n_samples, pattern, unknown, prefix):
    """ Create artificial bulk RNA-seq data from scRNA-seq dataset(s)"""
    simulation(simulate_dir=out,
               data_dir=data,
               sample_size=cells,
               num_samples=n_samples,
               pattern=pattern,
               unknown_celltypes=unknown,
               out_prefix=prefix)


"""
Generate example data
"""


@cli.command()
@click.option('--out',
              '-o',
              default='./',
              help="Directory to store output files in")
@click.option('--cells',
              '-c',
              default=10,
              help="Number of cells [default: 10]")
@click.option('--genes',
              '-g',
              default=100,
              help="Number of genes [default: 100]")
@click.option('--out',
              '-o',
              default="./",
              help="Output directory [default: ./]")
@click.option('--samples',
              '-n',
              default=10,
              help="Number of bulk samples [default: 10]")
def example(cells, genes, samples, out):
    exampleData(n_cells=cells, n_genes=genes, n_samples=samples, out_dir=out)
