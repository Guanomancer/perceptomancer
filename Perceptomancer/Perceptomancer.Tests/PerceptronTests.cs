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
    public class PerceptronTests
    {
        [Test]
        public void Train_Xor_Works()
        {
            var netDefinition = new int[] { 2, 3, 3, 1 };
            double learningRate = 0.3;
            double maxError = 0.01;
            int maxIterations = 30000;

            var perceptron = new Perceptron(netDefinition);
            var set = TraningSet.XorList();

            if (!perceptron.Learn(set.InputList, set.OutputList, learningRate, maxError, maxIterations, null, 10000))
            {
                Debug.WriteLine("Failed to train net.");
                return;
            }

            var result = new double[4];

            for (int index = 0; index < set.InputList.Count; index++)
            {
                var outputs = perceptron.Activate(set.InputList[index]);
                Debug.WriteLine($"Set #{index} produced:");
                for (int inp = 0; inp < set.InputList[index].Length; inp++)
                    Debug.Write($"  {Math.Round(set.InputList[index][inp], 2)}");
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
    }
}
