using System;

namespace Perceptomancer
{
    public interface IActivationFunction
    {
        double Activate(double value);
        double Derive(double value);
    }
}
