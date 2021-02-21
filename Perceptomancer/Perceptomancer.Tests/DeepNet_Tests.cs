using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Perceptomancer.Tests
{
    [TestFixture]
    public class DeepNet_Tests
    {
        [Test]
        public void Ctor_DoesNotFail()
        {
            var net = new DeepNet();
        }

        [Test]
        public void AddLayer_AddsLayerWithCorrentNumberOfInputs()
        {
            var net = new DeepNet();
            var inputLayer = new InputLayer(2);

            net.AddLayer(inputLayer);

            Assert.AreEqual(2, inputLayer.NumberOfInputs);
        }

        [Test]
        public void InputLayer_DoesNotChangeValues()
        {
            var net = new DeepNet();
            var inputLayer = new InputLayer(2);
            net.AddLayer(inputLayer);
            var result = net.Feed(new double[] { 0.75, 0.25 });

            Assert.AreEqual(0.75, result[0]);
            Assert.AreEqual(0.25, result[1]);
        }

        [Test]
        public void FullyconnectedLayer_ChangesValues()
        {
            var net = new DeepNet();
            net.AddLayer(new InputLayer(2));
            net.AddLayer(new FullyConnectedLayer(1));
            var result = net.Feed(new double[] { 0.75, 0.25 });

            Assert.AreNotEqual(0.75, result);
        }

        [Test]
        public void Train_Xor_ResultsInValidModel()
        {
            var traningSets = TraningSet.Xor();
            var net = new DeepNet();
            net.AddLayer(new InputLayer(2));
            net.AddLayer(new FullyConnectedLayer(3));
            net.AddLayer(new FullyConnectedLayer(3));
            net.AddLayer(new FullyConnectedLayer(1));

            var success = net.Train(traningSets, 0.3, 0.01, 10000);

            Assert.IsTrue(success);

            var result = new double[4];

            for (int index = 0; index < traningSets.Input.Count; index++)
            {
                var outputs = net.Feed(traningSets.Input[index]);
                Debug.WriteLine($"Set #{index} produced:");
                for (int inp = 0; inp < traningSets.Input[index].Length; inp++)
                    Debug.Write($"  {Math.Round(traningSets.Input[index][inp], 2)}");
                Debug.Write(" --> ");
                for (int op = 0; op < outputs.Length; op++)
                    Debug.Write($"  {Math.Round(outputs[op], 2)}");
                Debug.WriteLine(null);

                result[index] = outputs[0];
            }

            Assert.IsTrue(result[0] < 0.1);
            Assert.IsTrue(result[1] > 0.9);
            Assert.IsTrue(result[2] > 0.9);
            Assert.IsTrue(result[3] < 0.1);
        }

        [Test]
        public void Train_XorInverse_ResultsInValidModel()
        {
            var traningSets = TraningSet.XorInverse();
            var net = new DeepNet();
            net.AddLayer(new InputLayer(2));
            net.AddLayer(new FullyConnectedLayer(3));
            net.AddLayer(new FullyConnectedLayer(3));
            net.AddLayer(new FullyConnectedLayer(1));

            var success = net.Train(traningSets, 0.3, 0.01, 100000);

            Assert.IsTrue(success);

            var result = new double[4];

            for (int index = 0; index < traningSets.Input.Count; index++)
            {
                var outputs = net.Feed(traningSets.Input[index]);
                Debug.WriteLine($"Set #{index} produced:");
                for (int inp = 0; inp < traningSets.Input[index].Length; inp++)
                    Debug.Write($"  {Math.Round(traningSets.Input[index][inp], 2)}");
                Debug.Write(" --> ");
                for (int op = 0; op < outputs.Length; op++)
                    Debug.Write($"  {Math.Round(outputs[op], 2)}");
                Debug.WriteLine(null);

                result[index] = outputs[0];
            }

            Assert.IsTrue(result[0] > 0.9);
            Assert.IsTrue(result[1] < 0.1);
            Assert.IsTrue(result[2] < 0.1);
            Assert.IsTrue(result[3] > 0.9);
        }
    }
}
