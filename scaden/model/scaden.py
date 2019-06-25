"""
Cell Deconvolutional Network (scaden) class
"""
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import scanpy.api as sc
import collections
from .functions import dummy_labels, sample_scaling
from tqdm import tqdm

class Scaden(object):
    """
    scaden class
    """

    def __init__(self, sess, model_dir, model_name, batch_size=128, learning_rate=0.0001,  num_steps=1000):
        self.sess=sess
        self.model_dir=model_dir
        self.batch_size=batch_size
        self.model_name=model_name
        self.beta1=0.9
        self.beta2=0.999
        self.learning_rate=learning_rate
        self.data=None
        self.n_classes=None
        self.labels=None
        self.x=None
        self.y=None
        self.num_steps=num_steps
        self.scaling="log_min_max"
        self.sig_genes=None
        self.sample_names=None
        self.hidden_units = [256, 128, 64, 32]
        self.do_rates = [0, 0, 0, 0]


    def model_fn(self, X, n_classes, reuse=False):
        """
        Model function
        :param params:
        :param mode:
        :return:
        """
        activation = tf.nn.relu
        with tf.variable_scope("scaden_model", reuse=reuse):
            layer1 = tf.layers.dense(X, units=self.hidden_units[0], activation=activation , name="dense1")
            do1 = tf.layers.dropout(layer1, rate=self.do_rates[0], training=self.training_mode)
            layer2 = tf.layers.dense(do1, units=self.hidden_units[1], activation=activation , name="dense2")
            do2 = tf.layers.dropout(layer2, rate=self.do_rates[1], training=self.training_mode)
            layer3 = tf.layers.dense(do2, units=self.hidden_units[2], activation=activation , name="dense3")
            do3 = tf.layers.dropout(layer3, rate=self.do_rates[2], training=self.training_mode)
            layer4 = tf.layers.dense(do3, units=self.hidden_units[3], activation=activation , name="dense4")
            do4 = tf.layers.dropout(layer4, rate=self.do_rates[3], training=self.training_mode)
            logits = tf.layers.dense(do4, units=n_classes, activation=tf.nn.softmax, name="logits_layer")

            return logits

    def compute_loss(self, logits, targets):
        """
        Compute L1 loss
        :param logits:
        :param targets:
        :return: L1 loss
        """
        loss = tf.reduce_mean(np.square(logits - targets))
        return loss

    def compute_accuracy(self, logits, targets, pct_cut=0.05):
        """
        Compute prediction accuracy
        :param targets:
        :param pct_cut:
        :return:
        """
        equality = tf.less_equal(np.abs(np.subtract(logits, targets)), pct_cut)
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        return accuracy

    def correlation_coefficient(self, logits, targets):
        """
        Calculate the pearson correlation coefficient
        :param logits:
        :param targets:
        :return:
        """
        mx = tf.reduce_mean(logits)
        my = tf.reduce_mean(targets)
        xm, ym = logits-mx, targets-my
        r_num = tf.reduce_sum(tf.multiply(xm, ym))
        r_den = tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(xm)), tf.reduce_sum(tf.square(ym))))
        r = tf.divide(r_num, r_den)
        r = tf.maximum(tf.minimum(r, 1.0), -1.0)
        return r

    def visualization(self, logits, targets, classes):
        """
        Create evaluation metrics
        :param targets:
        :param classes:
        :return:
        """
        # add evaluation metrics
        rmse = tf.metrics.root_mean_squared_error(logits, targets)[1]
        pcor = self.correlation_coefficient(logits, targets)
        eval_metrics = {"rmse": rmse, "pcor": pcor}

        for i in range(logits.shape[1]):
            eval_metrics["mre_" + str(classes[i])] = tf.metrics.mean_relative_error(
                targets[:, i],
                logits[:, i],
                targets[:, i])[0]
            eval_metrics["mae_" + str(classes[i])] = tf.metrics.mean_absolute_error(
                targets[:, i],
                logits[:, i],
                targets[:, i])[0]
            eval_metrics["pcor_" + str(classes[i])] = self.correlation_coefficient(targets[:, i],logits[:, i])


        eval_metrics["mre_total"] = tf.metrics.mean_relative_error(targets,
                                                                   logits,
                                                                   targets)[1]

        eval_metrics["mae_total"] = tf.metrics.mean_relative_error(targets,
                                                                   logits,
                                                                   targets)[1]

        eval_metrics["accuracy01"] = self.compute_accuracy(logits, targets, pct_cut=0.01)
        eval_metrics["accuracy05"] = self.compute_accuracy(logits, targets, pct_cut=0.05)
        eval_metrics["accuracy1"] = self.compute_accuracy(logits, targets, pct_cut=0.1)

        # Create summary scalars
        for key, value in eval_metrics.items():
            tf.summary.scalar(key, value)

        tf.summary.scalar('loss', self.loss)

        merged_summary_op = tf.summary.merge_all()

        return merged_summary_op

    def load_h5ad_file(self, input_path, batch_size, datasets=[]):
        """
        Load input data from a h5ad file and divide into training and test set
        :param input_path: path to h5ad file
        :param batch_size: batch size to use for training
        :param datasets: a list of datasets to extract from the file
        :return: Dataset object
        """
        raw_input = sc.read_h5ad(input_path)

        # Subset dataset
        if len(datasets) > 0:
            all_ds = collections.Counter(raw_input.obs['ds'])
            for ds in all_ds:
                if ds not in datasets:
                    raw_input = raw_input[raw_input.obs['ds'] != ds].copy()


        # Create training dataset
        ratios = [raw_input.obs[ctype] for ctype in raw_input.uns['cell_types']]
        self.x_data = raw_input.X.astype(np.float32)
        self.y_data = np.array(ratios, dtype=np.float32).transpose()
        # create placeholders
        self.x_data_ph = tf.placeholder(self.x_data.dtype, self.x_data.shape, name="x_data_ph")
        self.y_data_ph = tf.placeholder(self.y_data.dtype, self.y_data.shape, name="y_data_ph")
        self.data = tf.data.Dataset.from_tensor_slices((self.x_data_ph, self.y_data_ph))
        self.data = self.data.shuffle(1000).repeat().batch(batch_size=batch_size)

        # Extract celltype and feature info
        self.labels = raw_input.uns['cell_types']
        self.sig_genes = list(raw_input.var_names)

    def load_prediction_file(self, input_path, sig_genes, labels, scaling=None):
        """
        Load a file to perform prediction on it
        :param input_path: path to input file
        :param sig_genes: the signature genes to use
        :param scaling: which scaling to perform
        :return: Dataset object
        """
        # Load data
        data = pd.read_table(input_path, sep="\t", index_col=0)
        sample_names = list(data.columns)

        # check for duplicates
        data_index = list(data.index)
        if not (len(data_index) == len(set(data_index))):
            print("Scaden Warning: Your mixture file conatins duplicate genes! The firs occuring gene will be used for every duplicate.")
            data = data.loc[~data.index.duplicated(keep='first')]

        data = data.loc[sig_genes]

        self.x_data = data.T
        self.x_data = self.x_data.astype(np.float32)
        m = self.x_data.shape[0]
        self.y_dummy = dummy_labels(m, labels)
        # Scaling
        if scaling:
            self.x_data = sample_scaling(self.x_data, scaling_option=scaling)

        # Create Dataset object from placeholders
        self.x_data_ph = tf.placeholder(self.x_data.dtype, self.x_data.shape, name="x_data_ph")
        self.y_data_ph = tf.placeholder(self.y_dummy.dtype, self.y_dummy.shape, name="y_data_ph")
        self.data = tf.data.Dataset.from_tensor_slices((self.x_data_ph, self.y_data_ph))
        self.data = self.data.batch(batch_size=m)

        return sample_names

    def build_model(self, input_path, train_datasets, mode="train"):
        """
        Build the model graph
        :param reuse:
        :return:
        """
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Load data
        if mode=="train":
            self.load_h5ad_file(input_path=input_path, batch_size=self.batch_size, datasets=train_datasets)

        if mode=="predict":
            self.sample_names = self.load_prediction_file(input_path=input_path, sig_genes=self.sig_genes,
                                                     labels=self.labels, scaling=self.scaling)

        # Make iterator
        iter = tf.data.Iterator.from_structure(self.data.output_types, self.data.output_shapes)
        next_element = iter.get_next()
        self.data_init_op = iter.make_initializer(self.data)
        self.x, self.y = next_element
        self.x = tf.cast(self.x, tf.float32)

        self.n_classes = len(self.labels)

        # Placeholder for training mode
        self.training_mode = tf.placeholder_with_default(True, shape=())

        # Model
        self.logits = self.model_fn(X=self.x, n_classes=self.n_classes)


        if mode == "train":
            # Loss
            self.loss = self.compute_loss(self.logits, self.y)
            # Summary scalars
            self.merged_summary_op = self.visualization(tf.cast(self.logits, tf.float32), targets=tf.cast(self.y, tf.float32), classes=self.labels)
            learning_rate = self.learning_rate
            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=self.global_step)


    def train(self, input_path, train_datasets):
        """
        Train the model
        :param num_steps:
        :return:
        """
        # Build model graph
        self.build_model(input_path=input_path, train_datasets=train_datasets, mode="train")

        # Init variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.saver = tf.train.Saver()
        model = os.path.join(self.model_dir, self.model_name)
        self.writer = tf.summary.FileWriter(model, self.sess.graph)
        self.eval_writer = tf.summary.FileWriter(os.path.join(self.model_dir, "eval"), self.sess.graph)

        # Initialize datasets
        self.sess.run(self.data_init_op, feed_dict={self.x_data_ph: self.x_data, self.y_data_ph: self.y_data})


        # Load pre-trained weights if avaialble
        self.load_weights(self.model_dir)

        # Training loop
        pbar = tqdm(range(self.num_steps))
        for _ in pbar:
            _, loss, summary = self.sess.run([self.optimizer, self.loss, self.merged_summary_op])
            self.writer.add_summary(summary, tf.train.global_step(self.sess, self.global_step))
            description = "Step: " + str(tf.train.global_step(self.sess, self.global_step)) + ", Loss: {:4.3f}".format(
                loss)
            pbar.set_description(desc=description)

        # Save the trained model
        self.saver.save(self.sess, model, global_step=self.global_step)
        # Save features and celltypes
        pd.DataFrame(self.labels).to_csv(self.model_dir + "/celltypes.txt", sep="\t")
        pd.DataFrame(self.sig_genes).to_csv(self.model_dir + "/genes.txt", sep="\t")


    def predict(self, input_path, out_name="cdn_predictions.txt"):
        """
        Perform prediction with a pre-trained model
        :param out_dir: path to store results in
        :param training_data: the dataset used for training
        :return:
        """
        # Load signature genes and celltype labels
        sig_genes = pd.read_table(self.model_dir + "/genes.txt", index_col=0)
        self.sig_genes = list(sig_genes['0'])
        labels = pd.read_table(self.model_dir + "/celltypes.txt", index_col=0)
        self.labels = list(labels['0'])

        # Build model graph
        self.build_model(input_path=input_path, train_datasets=[], mode="predict")

        # Initialize variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        self.saver = tf.train.Saver()

        model = os.path.join(self.model_dir, self.model_name)
        self.writer = tf.summary.FileWriter(model, self.sess.graph)

        # Initialize datasets
        self.sess.run(self.data_init_op, feed_dict={self.x_data_ph: self.x_data, self.y_data_ph: self.y_dummy})

        # Load pre-trained weights if avaialble
        self.load_weights(self.model_dir)

        predictions = self.sess.run([self.logits], feed_dict={self.training_mode: False})
        pred_df = pd.DataFrame(predictions[0], columns=self.labels, index=self.sample_names)
        #pred_df.to_csv(out_name, sep="\t")
        return pred_df

    def load_weights(self, model_dir):
            """
            Load pre-trained weights if available
            :param model_dir:
            :return:
            """
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Model parameters restored successfully")
