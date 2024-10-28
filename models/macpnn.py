import math
import pickle
import time
from typing import List
import numpy as np
from models.cpnn import cPNN


class SingleModel:
    def __init__(self, model_func=None, model=None, node_id=None):
        if model_func is not None:
            self.model = model_func()
        else:
            self.model = model
        if node_id is not None:
            self.node_id = node_id
        else:
            self.node_id = -1
        self.timestamp = 0


class MAcPNN:
    def __init__(
        self,
        model_func,
        num_batches: int = 50,
        initial_task: int = None,
        max_models: int = 10,
        local_models_prop: float = 0.4,
        handle_recursion: bool = False,
        training_steps_for_column_stability: int = 100
    ):
        """
        It implements a class to combine the knowledge of the current node and the one coming from external nodes.
        It receives the function to call to instantiate the first local model.
        All the models (external and locals) will be saved in the "models" attribute.
        When a new task i arrives, it adds a new column to all the models in "models" and builds an ensemble with them.
        After "num_batches" training mini-batches, it keeps in the ensemble only the model that performed better using
        the prequential evaluation. It then removes the last column to all the other models in "models".
        Before num_batches training mini-batches, predictions are made by the current task's best-performing model.
        In the case of supervised task labels (handle_recursion=True), for each already seen task, it saves in the
        "task_dict" attribute  the associated model and column.
        If more than one model has seen a task, it considers the model that performed better on that task.
        If an already seen task i arrives, it adds a new cPNN model to "models". The new model is obtained
        by considering a copy of the cPNN model associated with i and removing the columns after the one associated
        with i.

        Parameters
        ----------
        model_func: func.
            The function to be called to instantiate the first local model.
        num_batches: int, default: 50.
            Number of training mini-batches used to evaluate the best solution.
        initial_task: int, None.
            In the case of supervised task labels, the first task's label. Set to None in the case of no information
            about task labels.
        max_models: int, default: 10.
            The maximum number of models to be saved in the "models" attribute. After adding models from external nodes,
            if the current number of models exceeds max_models, only max_models models are kept. In this case the
            last ceil(local_models_prop * max_models) local models will be selected by sorting them using the last usage
            timestamp. The remaining models will be chosen from the external models by sorting them using the
            last usage timestamp.
        local_models_prop: float, default: 0.4.
            After adding models from external nodes, if the current number of models exceeds max_models, it represents
            the proportion of local models' number with respect to max_models.
        handle_recursion: bool, default: False.
            Set it to True if the model must handle supervised tasks' labels.
        training_steps_for_column_stability: int, default: 100
            Number of training steps to consider stable the last column. If an external node requests for the local
            models, the last column will be sent only if it has been training for at least
            training_steps_for_column_stability training mini-batches.

        Attributes
        ----------
        self.models: list[SingleModel]
            It contains local and external models. Each element is an object with the following attributes:
            - model: the cPNN model
            - node_id: the id of the node to which the model belongs (-1 if the model is local)
            - timestamp: the timestamp of the model's last selection
        """
        self.models: List[SingleModel] = [SingleModel(model_func=model_func)]
        self._current_models: List[cPNN] = []
        self.cont = 0
        self.num_data_points = num_batches * self.models[0].model.batch_size
        self._current_selection = 0
        self._current_task = 0
        self._already_seen_task = False
        self.max_models = max_models
        self.local_model_prop = local_models_prop
        self.n_local_models = math.ceil(self.local_model_prop * self.max_models)
        self.handle_recursion = handle_recursion
        if initial_task is None:
            initial_task = 1
        self.task_dict = {}
        self._add_column = False
        self.add_new_column(initial_task)
        self.training_steps_for_column_stability = training_steps_for_column_stability

    def add_models(self, models: List[SingleModel], node_id: int):
        """
        It adds to the ensemble the updated models of an external node.
        Before adding the updated models of the external nodes, it removes all the models with that node_id.
        After adding the models, if the current number of models exceeds max_models, only max_models models are
        kept. In this case, the last ceil(local_models_prop * max_models) local models will be chosen by sorting them
        using the last usage timestamp. The remaining models will be chosen from the external models by sorting them
        using the last usage timestamp.

        Parameters
        ----------
        models: list of SingleModel
            The cPNN models learned by the external node. It's a list in which each element is an object
            with the following attributes:
            - model: the cPNN model
            - node_id: the id of the node to which the model belongs (-1 if the model is local)
            - timestamp: the timestamp of the model's last selection
        node_id: int
            The id of the external node
        Returns
        -------

        """
        i = -1
        for m in models:
            m.node_id = node_id
        self.models += models

        self.models = sorted(self.models, key=lambda m: m.timestamp, reverse=True)
        if len(self.models) > self.max_models:
            local_models = [m for m in self.models if m.node_id == -1]
            n_local_models = min(len(local_models), self.n_local_models)
            local_models = local_models[:n_local_models]
            n_external_models = self.max_models - n_local_models
            external_models = [m for m in self.models if m.node_id != -1][
                :n_external_models
            ]
            self.models = local_models + external_models

        self.task_dict = {}
        if self.handle_recursion:
            for i, model in enumerate(self.models):
                for col, task in enumerate(model.model.task_ids):
                    if (
                        task not in self.task_dict
                        or model.model.columns_perf[col].get()
                        > self.task_dict[task]["perf"]
                    ):
                        self.task_dict[task] = {
                            "model": i,
                            "column": col,
                            "perf": model.model.columns_perf[col].get(),
                        }

    def add_new_column(self, task_id: int = None):
        """
        When a drift arises, call this method.

        Parameters
        ----------
        task_id: int, default: None.
            The id of the task following the drift. It could be also an already seen task.
            If None, it adds 1 to the maximum id of the already seen tasks.
        Returns
        -------

        """
        if 0 < self.cont < self.num_data_points:
            self._select_model()

        if task_id is None:
            if len(self.task_dict) == 0:
                self._current_task = 1
            else:
                self._current_task = max(list(self.task_dict.keys())) + 1
        else:
            self._current_task = task_id

        if task_id not in self.task_dict:
            self._current_models = [m.model for m in self.models]
            if self._add_column:
                for model in self._current_models:
                    model.add_new_column(task_id)
            if self.handle_recursion:
                self.task_dict[task_id] = {}
            self.cont = 0
            self._already_seen_task = False
        else:
            model: cPNN = self.models[self.task_dict[task_id]["model"]].model
            model = pickle.loads(pickle.dumps(model))
            model.take_first_columns(self.task_dict[task_id]["column"] + 1)
            model.unfreeze_last_column()
            self._current_models = [m.model for m in self.models]
            if self._add_column:
                for model_ in self._current_models:
                    model_.add_new_column(task_id)
            self.cont = 0
            self._current_models += [model]
            self._already_seen_task = True

        self._current_selection = np.random.choice(
            list(range(len(self._current_models)))
        )
        self._add_column = True

    def _select_model(self):
        if (
            self._already_seen_task
            and self._current_selection == len(self._current_models) - 1
        ):
            self.models.append(
                SingleModel(model=self._current_models[self._current_selection])
            )
        for i, model in enumerate(self._current_models):
            if i != self._current_selection:
                model.remove_last_column()
        if self.models[self._current_selection].node_id != -1:
            self.models[self._current_selection].node_id = -1
        self.models[self._current_selection].timestamp = time.time()
        self.models = [m for m in self.models if m.node_id == -1]
        self._local_models_to_add = []
        if self.handle_recursion:
            self.task_dict[self._current_task]["model"] = self._current_selection
            self.task_dict[self._current_task]["column"] = (
                len(self._current_models[self._current_selection].columns.columns) - 1
            )
        self._current_models = [self._current_models[self._current_selection]]
        self._current_selection = 0
        self._already_seen_task = False

    def learn_one(self, x: np.array, y: int):
        """
        It trains the models on a single data point.
        Before num_batches batches of the current task, it trains an ensemble of cPNN models.
        After num_batches batches it keeps only the best-performing cPNN model on the current task.

        Parameters
        ----------
        x: numpy.array or list
            The features values of the data point.
        y: int
            The target values of the data point.

        Returns
        -------

        """
        for model in self._current_models:
            model.learn_one(x, y)

        if self.cont < self.num_data_points:
            self.cont += 1
            self._current_selection = np.argmax(
                [model.model.columns_perf[-1].get() for model in self.models]
            )
            if self.cont == self.num_data_points:
                self._select_model()

        if self.handle_recursion:
            self.task_dict[self._current_task]["perf"] = (
                self._current_models[self._current_selection].columns_perf[-1].get()
            )

    def predict_one(
        self, x: np.array, column_id: int = None, previous_data_points: np.array = None
    ):
        """
        It performs prediction on a single data point.
        Before num_batches batches of the current task, it returns the predictions of the ensemble's best-performing
        cPNN on the current task.

        Parameters
        ----------
        x: numpy.array or list
            The features values of the single data point.
        column_id: int, default: None.
            The id of the column to use. If None the last column is used.
        previous_data_points: numpy.array, default: None.
            The features value of the data points preceding x in the sequence.
            If None, it uses the last seq_len-1 points seen during the last calls of the method.
            It returns None if the model has not seen yet seq_len-1 data points and previous_data_points is None.
        Returns
        -------
        prediction : int
            The predicted int label of x.
        """
        predictions = []
        for i, model in enumerate(self._current_models):
            predictions.append(
                model.predict_one(
                    x, column_id=column_id, previous_data_points=previous_data_points
                )
            )
        return predictions[self._current_selection]

    def get_seq_len(self):
        return self._current_models[self._current_selection].get_seq_len()

    def send_local_models(self):
        """
        It returns a list containing copies of the local models. Use this method to add the local models of the
        current F-cPNN to another F-cPNN.

        Returns
        -------
        List[SingleModel]: The list of the local cPNN models. Each element is an object with the following attributes:
            - model: the cPNN model
            - node_id: the id of the node to which the model belongs (-1)
            - timestamp: the timestamp of the model's last selection
        """
        models = []
        for m in self.models:
            if m.node_id == -1:
                if m.model.train_cont[-1] >= self.training_steps_for_column_stability:
                    models.append(pickle.loads(pickle.dumps(m)))
                elif len(m.model.columns.columns) > 1:
                    m = pickle.loads(pickle.dumps(m))
                    m.model.remove_last_column()
                    models.append(m)
        return models
