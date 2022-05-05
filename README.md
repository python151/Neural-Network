# Neural-Network
A Basic ANN implementation using Python, Numpy, and the NetworkX graph library. Currently it exclusively uses a genetic algorithm to train, but that will change in the future.

# Usage
It's essentially just a reimplementation of a generic neural network for ML that employs the use of a genetic algorithm (will likely expand training methods in the future).
The program has some options for configury, but for the most part they are coded directly in so, in order to use them, you need to go and change some variable in the script itself.

<b>Configuration:</b>
<ul>
  <li>You can change the <code>num_hidden_layers</code>, <code>layer_size</code>, <code>input_shape</code>, and <code>output_shape</code> through parameters in the constructor of the Network class.</li>
  <li>You can change the <code>generation_size</code>, <code>num_generations</code>, <code>filter_size</code> (Parameters of the genetic algorithm) in the section labeled 'Configuring training' of <code>main.py</code>.</li>
  <li>You can change up the data itself in the area of <code>main.py</code> labeled 'Loading in Dataset' (will likely automate process soon).</li>
</ul>

<b>Scripts + Analytics:</b>
<ul>
  <li>The <code>main.py</code> script will train and store the model to a file.</li>
  <li>The <code>Draw_network.py</code> script will write a visual graph-based representation of the model into a .png file.</li>
  <li>The <code>Display_metrics</code> script will display the time complexity, and plot some graphs to show the models performance.</li>
</ul>
