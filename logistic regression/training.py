#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

from typing import List
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from optimizers import Param
from model import compute_output, compute_cost, compute_gradient, compute_hessian
from optimizers import LineSearchOptimizer, NewtonsMethodOptimizer
from collections import namedtuple
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from pynput import keyboard
import threading

# A record for a training step
Record = namedtuple('Record', ['params', 'cost', 'accuracy', 'precision', 'recall', 'f1score'])


# Helper class to generate 1D training animation provided you pass proper training data
class TrainingAnimation1D:
  def __init__(self, history: List[Record], predictors: np.ndarray, responses: np.ndarray,
               max_roots=5,
               tracked_metrics=None):
    self.history = history
    self.predictors = predictors
    self.responses = responses
    self.max_roots = max_roots
    self.boundary_artists = None
    self.cost_artists = None
    self.metrics_artists = None
    if tracked_metrics is None:
      self.tracked_metrics = ['accuracy', 'precision', 'recall', 'f1score']
    else:
      self.tracked_metrics = tracked_metrics
    self.current = 0
    self.lock = threading.Lock()

  # Setups metrics axis returning a dictionary with an artist entry per metric
  def _setup_metrics_artists(self, ax):
    artists = {}
    for entry in self.tracked_metrics:
      artists[entry] = ax.plot([], [], '.-', label=entry, alpha=0.5)[0]
    ax.set_title('Metrics')
    ax.set_xlabel('Step')
    ax.set_xlim([0, len(self.history)])
    ax.set_ylim([0, 1])
    ax.grid()
    return artists

  # Setups boundary axis returning the axis
  def _setup_boundary_artists(self, ax):
    ax.scatter(self.predictors[:, 1], self.responses, c=self.responses, cmap='viridis', s=40, edgecolor='k')
    artists = []
    for i in range(self.max_roots):
      line = ax.plot([], [], 'k-')[0]
      artists.append(line)
    ax.set_title('Boundary Fit')
    ax.set_xlabel('$x$')
    ax.grid()
    return artists

  # Setups cost axis and returns artist
  def _setup_cost_artists(self, ax):
    line = ax.plot([], [], 'k.-')[0]
    ax.set_title('Cost function $J(w)$')
    ax.set_xlabel('Step')
    ax.grid()
    ax.set_xlim([0, len(self.history)])
    ax.set_ylim([0, max([entry[1] for entry in self.history])])
    return line

  # Updates data of artists from history step
  def _plot_frame(self, step):
    if step < len(self.history):
      # Boundary Fit
      ydata = [0, 1]
      weights = self.history[step].params
      roots = np.roots(weights[::-1])
      for i in range(min(len(roots), self.max_roots)):
        xdata = [roots[i], roots[i]]
        self.boundary_artists[i].set_data(xdata, ydata)
      # Cost
      xdata = range(0, step + 1)
      ydata = [record.cost for record in self.history[:step + 1]]
      self.cost_artists.set_data(xdata, ydata)
      # Metrics
      for key, artist in self.metrics_artists.items():
        ydata = [getattr(record, key) for record in self.history[:step + 1]]
        artist.set_data(range(0, step + 1), ydata)

  # Keyboard listener
  def stop(self, key):
    if key == keyboard.Key.esc:
      with self.lock:
        self.current = len(self.history) - 1

  # Starts animation
  def start(self):
    # Creates figure
    fig = plt.figure(figsize=(17, 4))
    fig.subplots_adjust(top=1.2, bottom=0.4)
    # Creates axes
    ax_boundary = fig.add_subplot(131)
    ax_cost = fig.add_subplot(132)
    ax_metrics = fig.add_subplot(133)
    # Plots static data and gets reference for updating data
    self.boundary_artists = self._setup_boundary_artists(ax_boundary)
    self.cost_artists = self._setup_cost_artists(ax_cost)
    self.metrics_artists = self._setup_metrics_artists(ax_metrics)
    # Performs animation by updating data
    clear_output(wait=True)
    plt.pause(0.01)
    self._plot_frame(self.current)
    fig.legend(loc=8)
    # Keyboard listener to stop animation
    listener = keyboard.Listener(on_press=self.stop)
    listener.start()
    # Looping through steps
    while self.current < len(self.history):
      with self.lock:
        self._plot_frame(self.current)
        display(fig)
        clear_output(wait=True)
        self.current += 1
      plt.pause(0.1)


# Helper class to generate 2D training animation provided you pass proper training data
class TrainingAnimation2D:
  def __init__(self, history: List[Record], predictors: np.ndarray, responses: np.ndarray,
               polynomial=None,
               scaler=None,
               tracked_metrics=None,
               threshold=0.5):
    self.history = history
    self.predictors = predictors
    self.responses = responses
    self.polynomial = polynomial
    self.scaler = scaler
    self.threshold = threshold
    self.boundary_axis = None
    self.boundary_mesh = None
    self.cost_artists = None
    self.metrics_artists = None
    if tracked_metrics is None:
      self.tracked_metrics = ['accuracy', 'precision', 'recall', 'f1score']
    else:
      self.tracked_metrics = tracked_metrics
    self.current = 0
    self.lock = threading.Lock()

  # Creates mesh applying same transformation
  def _setup_boundary_mesh(self):
    xlim = (np.array(self.boundary_axis.get_xlim()) * np.sqrt(self.scaler.var_[0])) + self.scaler.mean_[0]
    ylim = (np.array(self.boundary_axis.get_ylim()) * np.sqrt(self.scaler.var_[1])) + self.scaler.mean_[1]
    pts = 200
    mx = np.linspace(xlim[0], xlim[1], pts)  # x1
    my = np.linspace(ylim[0], ylim[1], pts)  # x2
    mx, my = np.meshgrid(mx, my)
    mesh = np.vstack((mx.flatten(), my.flatten())).T
    if self.polynomial is not None:
      mesh = self.polynomial.transform(mesh)
    if self.scaler is not None:
      mesh[:, 1:] = self.scaler.transform(mesh[:, 1:])
    return mesh

  # Setups boundary axis returning the axis
  def _setup_boundary_axis(self, ax):
    ax.scatter(self.predictors[:, 1], self.predictors[:, 2], c=self.responses, cmap='viridis', s=40, edgecolor='k')
    ax.set_title('Boundary Fit')
    ax.set_xlabel('$x$')
    return ax

  def _update_boundary_axis(self, step):
    self.boundary_axis.cla()
    self.boundary_axis.scatter(self.predictors[:, 1], self.predictors[:, 2],
                               c=self.responses, cmap='viridis', s=40, edgecolor='k')
    pts = int(np.sqrt(self.boundary_mesh.shape[0]))
    x1 = self.boundary_mesh[:, 1].reshape((pts, pts))
    x2 = self.boundary_mesh[:, 2].reshape((pts, pts))
    h = compute_output(self.boundary_mesh, self.history[step].params)
    z = h.reshape((pts, pts))
    self.boundary_axis.contour(x1, x2, z, levels=[self.threshold], colors='black')
    self.boundary_axis.contourf(x1, x2, z >= self.threshold, alpha=0.10, cmap='viridis')

  # Setups metrics axis returning a dictionary with an artist entry per metric
  def _setup_metrics_artists(self, ax):
    artists = {}
    for entry in self.tracked_metrics:
      artists[entry] = ax.plot([], [], '.-', label=entry, alpha=0.5)[0]
    ax.set_title('Metrics')
    ax.set_xlabel('Step')
    ax.set_xlim([0, len(self.history)])
    ax.set_ylim([0, 1])
    ax.grid()
    return artists

  # Setups cost axis and returns artist
  def _setup_cost_artists(self, ax):
    line = ax.plot([], [], 'k.-')[0]
    ax.set_title('Cost function $J(w)$')
    ax.set_xlabel('Step')
    ax.grid()
    ax.set_xlim([0, len(self.history)])
    ax.set_ylim([0, max([entry[1] for entry in self.history])])
    return line

  # Keyboard listener
  def stop(self, key):
    if key == keyboard.Key.esc:
      with self.lock:
        self.current = len(self.history) - 1

  # Starts animation
  def start(self):
    # Creates figure
    fig = plt.figure(figsize=(17, 4))
    fig.subplots_adjust(top=1.2, bottom=0.4)
    # Creates axes
    ax_boundary = fig.add_subplot(131)
    ax_cost = fig.add_subplot(132)
    ax_metrics = fig.add_subplot(133)
    # Plots static data and gets reference for updating data
    self.boundary_axis = self._setup_boundary_axis(ax_boundary)
    self.boundary_mesh = self._setup_boundary_mesh()
    self.cost_artists = self._setup_cost_artists(ax_cost)
    self.metrics_artists = self._setup_metrics_artists(ax_metrics)
    # Performs animation by updating data
    clear_output(wait=True)
    plt.pause(0.01)
    self._plot_frame(self.current)
    fig.legend(loc=8)
    # Keyboard listener to stop animation
    listener = keyboard.Listener(on_press=self.stop)
    listener.start()
    # Looping through steps
    while self.current < len(self.history):
      with self.lock:
        self._plot_frame(self.current)
        display(fig)
        clear_output(wait=True)
        self.current += 1
      plt.pause(0.1)

  # Updates data of artists from history step
  def _plot_frame(self, step):
    if step < len(self.history):
      # Boundary Fit
      self._update_boundary_axis(step)
      # Cost
      xdata = range(0, step + 1)
      ydata = [record.cost for record in self.history[:step + 1]]
      self.cost_artists.set_data(xdata, ydata)
      # Metrics
      for key, artist in self.metrics_artists.items():
        ydata = [getattr(record, key) for record in self.history[:step + 1]]
        artist.set_data(range(0, step + 1), ydata)


# Training loop!
def train(predictors, responses, starting_point, optimizer,
          epochs=200,
          batch_size=None,
          steps_per_epoch=None,
          epsilon=1e-5,
          early_stopping_steps=10,
          threshold=0.5) -> List[Record]:
  # Number of observations
  m = predictors.shape[0]
  # Creates weights param
  weights = Param((predictors.shape[1],))
  weights.value = np.array(starting_point, dtype=np.float32)
  # How many batches?
  if batch_size is None:
    steps_per_epoch = 1
    num_batches = None
  else:
    num_batches = int(np.ceil(m / batch_size))
    if steps_per_epoch is None:
      steps_per_epoch = num_batches
  # We store the results of the trainings here
  optimizer.reset([weights])
  # Computes and stores the cost for the initial weights
  history = []
  h = compute_output(predictors, weights.value)
  J = compute_cost(responses, h)
  # Place record on history
  estimations = (h >= threshold)
  record = Record(
    params=weights.value.copy(),
    cost=J,
    accuracy=accuracy_score(responses, estimations),
    precision=precision_score(responses, estimations),
    recall=recall_score(responses, estimations),
    f1score=f1_score(responses, estimations)
  )
  history.append(record)
  # Training loop
  stop = False
  badsteps = 0
  for epoch in range(epochs):
    # Shuffles indexes of observation to generate random batches
    batch_indexes = np.random.permutation(m)
    # Performs steps for this epoch
    for step in range(steps_per_epoch):
      # Determines gradient of weights
      if batch_size is None:
        h = compute_output(predictors, weights.value)
        grad = compute_gradient(predictors, responses, h)
      else:
        start = step % num_batches
        samples = batch_indexes[start * batch_size: (start + 1) * batch_size]
        h = compute_output(predictors[samples, :], weights.value)
        grad = compute_gradient(predictors[samples, :], responses[samples], h)

      # Updates weights
      if isinstance(optimizer, (LineSearchOptimizer, NewtonsMethodOptimizer)):
        optimizer.update([weights], [grad], [compute_hessian(predictors, h)])
      else:
        optimizer.update([weights], [grad])

      # Computes new cost after weights update
      h = compute_output(predictors, weights.value)
      J = compute_cost(responses, h)

      # Check if we the cost has gone crazy!
      if np.isinf(J) or np.isnan(J):
        stop = True
        break

      # Place record on history
      estimations = (h >= threshold)
      record = Record(
        params=weights.value.copy(),
        cost=J,
        accuracy=accuracy_score(responses, estimations),
        precision=precision_score(responses, estimations),
        recall=recall_score(responses, estimations),
        f1score=f1_score(responses, estimations)
      )
      history.append(record)
      delta = history[-2].cost - history[-1].cost
      # Check whether cost function changed enough
      if abs(delta) < epsilon:
        stop = True
        break
      # Early stopping
      badsteps = badsteps + 1 if delta < 0 else 0
      if badsteps > early_stopping_steps:
        stop = True
        break
        # Any exit condition met?
    if stop:
      break

  # Outputs results
  return history
