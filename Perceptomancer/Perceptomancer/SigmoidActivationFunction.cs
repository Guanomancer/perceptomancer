using System;

namespace Perceptomancer
{
    public class SigmoidActivationFunction : IActivationFunction
    {
        public static SigmoidActivationFunction Default;

        static SigmoidActivationFunction() => Default = new SigmoidActivationFunction();

        public double Activate(double value) => 1 / (1 + Math.Exp(-value));

        public double Derive(double value)
        {
            double y = Activate(value);
            return y * (1 - y);
        }
    }
}
