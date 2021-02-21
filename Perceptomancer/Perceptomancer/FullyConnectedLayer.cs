using System;

namespace Perceptomancer
{
    public class FullyConnectedLayer : DeepLayer
    {
        public FullyConnectedLayer(int numberOfNeurons, IActivationFunction activationFunction)
            : base(numberOfNeurons, activationFunction)
        { }

        public override void Attach(DeepLayer previousLayer = null)
        {
            if (previousLayer == null)
                throw new ArgumentNullException("previousLayer");

            NumberOfInputs = previousLayer.NumberOfNeurons;
            for (int i = 0; i < NumberOfNeurons; i++)
                Neurons.Add(new DeepNeuron(NumberOfInputs, ActivationFunction, RandomNumberGenerator));
        }

        public override double[] Feed(double[] inputValues)
        {
            for(int i = 0; i < NumberOfNeurons; i++)
                Output[i] = Neurons[i].Activate(inputValues);

            return Output;
        }
    }
}
