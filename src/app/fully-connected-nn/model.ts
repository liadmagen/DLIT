import * as dl from 'deeplearn';
import {Graph, Optimizer, Session, SGDOptimizer, Tensor} from 'deeplearn';
import {InCPUMemoryShuffledInputProviderBuilder} from 'deeplearn/dist/data/input_provider';
import {NDArrayMath} from 'deeplearn/dist/math/math';

// const math = ENV.math;
// const a = Array1D.new([1, 2, 3]);
// const b = Scalar.new(2);

// const result = math.add(a, b);

// result.data().then(data => console.log(data));

export class Model {
  readonly IMAGE_SIZE = 28;
  readonly IMAGE_PIXELS = this.IMAGE_SIZE * this.IMAGE_SIZE;
  readonly NUM_CLASSES = 10;
  readonly BATCH_SIZE = 64;

  math: NDArrayMath = dl.ENV.math;
  graph: Graph;
  initialLearningRate: number;
  optimizer: Optimizer;
  session: Session;

  inputTensor: Tensor;
  targetTensor: Tensor;

  predictionTensor: Tensor;
  costTensor: Tensor;

  feedEntries = [];

  constructor(initialLearningRate: number = 0.1) {
    this.graph = new Graph();
    this.initialLearningRate = initialLearningRate;
    this.optimizer = new SGDOptimizer(this.initialLearningRate);
  }

  placeholderInputs(batchSize) {
    this.inputTensor =
        this.graph.placeholder('input MNIST', [batchSize, this.IMAGE_PIXELS]);
    this.targetTensor =
        this.graph.placeholder('output digit', [batchSize, this.NUM_CLASSES]);

    return [this.inputTensor, this.targetTensor];
  }

  createFullyConnectedLayer(
      inputLayer: dl.Tensor, layerIndex: number, sizeOfThisLayer: number,
      useBias: boolean = true): Tensor {
    return this.graph.layers.dense(
        'fully_connected_' + layerIndex, inputLayer, sizeOfThisLayer,
        (x) => this.graph.relu(x), useBias);
  }

  /**
 * By default, creates a model with 4 hidden layers:
 * 64 nodes
 * 32 nodes
 * 16 nodes
 * 10 nodes
 * RELU, Softmax at the end
 * loss function: euclidean distance
 */
  buildGraph() {
    [this.inputTensor, this.targetTensor] =
        this.placeholderInputs(this.BATCH_SIZE);

    // add layers to the graph
    let fullyConnectedLayer: Tensor =
        this.createFullyConnectedLayer(this.inputTensor, 0, 64);

    fullyConnectedLayer =
        this.createFullyConnectedLayer(fullyConnectedLayer, 1, 32);

    fullyConnectedLayer =
        this.createFullyConnectedLayer(fullyConnectedLayer, 2, 16);

    fullyConnectedLayer =
        this.createFullyConnectedLayer(fullyConnectedLayer, 3, 10);

    this.predictionTensor = this.graph.softmax(fullyConnectedLayer);

    this.costTensor =
        this.graph.meanSquaredCost(this.targetTensor, this.predictionTensor);

    this.session = new Session(this.graph, this.math);

    return this.session;
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

  trainBatch() {
    const learningRate =
        this.initialLearningRate * Math.pow(0.85, Math.floor(step / 42));
    (this.optimizer as SGDOptimizer).setLearningRate(learningRate);

    let costValue = -1;
    this.math.scope(() => {
      const cost = this.session.train(
          this.costTensor, this.feedEntries, this.BATCH_SIZE, this.optimizer,
          dl.CostReduction.MEAN);

      costValue = cost.get();
    });

    return costValue;
  }

  train(inputArray, targetArray) {
    const shuffledInputProviderBuilder =
        new InCPUMemoryShuffledInputProviderBuilder([inputArray, targetArray]);
    const [inputProvider, targetProvider] =
        shuffledInputProviderBuilder.getInputProviders();

    this.feedEntries = [
      {tensor: this.inputTensor, data: inputProvider},
      {tensor: this.targetTensor, data: targetProvider}
    ];

    const epochs = 100;
    for (let step = 1; step < epochs; step++) {
      this.trainBatch();
    }
  }

  /*
  infer(x: dl.Array1D, vars: {[varName: string]: dl.NDArray}): dl.Scalar {
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
  }*/
}
