import * as dl from 'deeplearn';
import {Graph, Optimizer, Session, SGDOptimizer, Tensor} from 'deeplearn';
import {InCPUMemoryShuffledInputProviderBuilder} from 'deeplearn/dist/data/input_provider';

// const math = ENV.math;
// const a = Array1D.new([1, 2, 3]);
// const b = Scalar.new(2);

// const result = math.add(a, b);

// result.data().then(data => console.log(data));

/**
 * By default, creates a model with 6 layers:
 * 8 nodes
 * 7 nodes
 * 6 nodes
 * 5 nodes
 * 4 nodes
 * 2 nodes
 * RELU, Softmax at the end
 * loss function: euclidean distance
 */

export class Model {
  IMAGE_SIZE = 28;
  IMAGE_PIXELS = this.IMAGE_SIZE * this.IMAGE_SIZE;
  NUM_CLASSES = 10;
  BATCH_SIZE = 64;

  math = dl.ENV.math;
  graph = new Graph();
  initialLearningRate: number;
  optimizer: SGDOptimizer;
  session: Session;
  inputTensor: Tensor;
  targetTensor: Tensor;
  predictionTensor: Tensor;
  costTensor: Tensor;

  constructor(initialLearningRate: number = 0.1) {
    this.initialLearningRate = initialLearningRate;
    this.optimizer = new SGDOptimizer(this.initialLearningRate);
  }

  placeholder_inputs(batch_size) {
    this.inputTensor =
        this.graph.placeholder('input MNIST', [batch_size, this.IMAGE_PIXELS]);
    this.targetTensor =
        this.graph.placeholder('output digit', [batch_size, this.NUM_CLASSES]);

    return [this.inputTensor, this.targetTensor];
  }

  createFullyConnectedLayer(
      graph: dl.Graph, inputLayer: dl.Tensor, layerIndex: number,
      sizeOfThisLayer: number, includeBias: boolean = true) {
    return graph.layers.dense(
        'fully_connected_' + layerIndex, inputLayer, sizeOfThisLayer,
        (x) => graph.relu(x), includeBias);
  }

  buildGraph() {
    [this.inputTensor, this.targetTensor] =
        this.placeholder_inputs(this.BATCH_SIZE);
    let fullyConnectedLayer =
        this.createFullyConnectedLayer(this.graph, this.inputTensor, 0, 64);

    fullyConnectedLayer =
        this.createFullyConnectedLayer(this.graph, fullyConnectedLayer, 1, 32);

    fullyConnectedLayer =
        this.createFullyConnectedLayer(this.graph, fullyConnectedLayer, 2, 16);

    this.predictionTensor =
        this.createFullyConnectedLayer(this.graph, fullyConnectedLayer, 3, 3);

    this.costTensor =
        this.graph.meanSquaredCost(this.targetTensor, this.predictionTensor);

    this.session = new dl.Session(this.graph, this.math);
  }

  renderMnistImage(array: dl.Array1D) {
    const width = 28;
    const height = 28;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const float32Array = array.dataSync();
    const imageData = ctx.createImageData(width, height);
    for (let i = 0; i < float32Array.length; ++i) {
      const j = i * 4;
      const value = Math.round(float32Array[i] * 255);
      imageData.data[j + 0] = value;
      imageData.data[j + 1] = value;
      imageData.data[j + 2] = value;
      imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
    return canvas;
  }

  train() {
    const learningRate =
        this.initialLearningRate * Math.pow(0.85, Math.floor(step / 42));
    this.optimizer.setLearningRate(learningRate);

    let costValue = -1;
    this.math.scope(() => {
      const cost = this.session.train(
          this.costTensor, this.feedEntries, this.BATCH_SIZE, this.optimizer,
          dl.CostReduction.MEAN);

      costValue = cost.get();
    });

    return costValue;
  }

  public train(inputArray, targetArray) {
    const shuffledInputProviderBuilder =
        new InCPUMemoryShuffledInputProviderBuilder([inputArray, targetArray]);
    const [input_provider, targetProvider] =
        shuffledInputProviderBuilder.getInputProviders();

    this.feedEntries = [
      {tensor: this.inputTensor, data: inputProvider},
      {tensor: this.targetTensor, data: targetProvider}
    ];
  }

  public infer(x: dl.Array1D, vars: {[varName: string]: dl.NDArray}):
      dl.Scalar {
    const hidden1W = vars['hidden1/weights'] as dl.Array2D;
    const hidden1B = vars['hidden1/biases'] as dl.Array1D;
    const hidden2W = vars['hidden2/weights'] as dl.Array2D;
    const hidden2B = vars['hidden2/biases'] as dl.Array1D;
    const softmaxW = vars['softmax_linear/weights'] as dl.Array2D;
    const softmaxB = vars['softmax_linear/biases'] as dl.Array1D;

    const hidden1 =
        x.as2D(-1, hidden1W.shape[0]).matMul(hidden1W).add(hidden1B).relu() as
        dl.Array1D;
    const hidden2 = hidden1.as2D(-1, hidden2W.shape[0])
                        .matMul(hidden2W)
                        .add(hidden2B)
                        .relu() as dl.Array1D;
    const logits =
        hidden2.as2D(-1, softmaxW.shape[0]).matMul(softmaxW).add(softmaxB);

    return logits.argMax();
  }
}
