import { Ihyperparameters } from '../ihyperparameters';

export class FullyConnectedNnHyperparmeters implements Ihyperparameters {
    public model_type = 'fully_connected_network';
    public training_set = '';
    public loss_function = '';
    public learning_rate = 0.1;
    public model = [];
}
