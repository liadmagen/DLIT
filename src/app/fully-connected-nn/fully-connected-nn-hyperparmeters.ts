import { Ihyperparameters } from '../ihyperparameters';
import {Optimizer, SGDOptimizer, Tensor} from 'deeplearn';

export class FullyConnectedNnHyperparmeters implements Ihyperparameters {
    public modelType = 'fully_connected_network';
    public normalization = '';
    public dataset = '';
    public lossFunction = '';
    public learningRate = 0.1;
    public optimizer: Optimizer = new SGDOptimizer(this.learningRate);
    public layers = [];
}
