import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.net.DatagramSocket;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

import com.sun.org.apache.xerces.internal.dom.DeferredCDATASectionImpl;

import sun.print.resources.serviceui;

public class Perceptron {
	
	// Predict which class the instance belongs to (0 or 1)
	// inputs is a 6 dimensional vector and weights is a 6 dimensional vector
	public static int predict(double[] inputs, double[] weights) {
		double activation = weights[0]; // First value of weights is the bias
		// Calculate dot product of weights and inputs (excluding the last value of inputs because it is the expected output value)
		for (int i = 0; i < inputs.length - 1; i++) {
			activation += weights[i+1] * inputs[i];
		}
		if (activation >= 0) {
			return 1;
		}
		return 0;
	}
	
	public static double[] updateWeights(ArrayList<double[]> dataSet, double learningRate, int numOfEpochs) {
		// Initialize the weight vector to be all zeros
		double[] weights = new double[dataSet.get(0).length];
		for (int i = 0; i < dataSet.get(0).length - 1; i++) {
			weights[i] = 0;
		}
		// Calculate the summation of the errors for each set of inputs
		for (int i = 0; i < numOfEpochs; i++) {
			double sumError = 0;
			for (double[] inputs : dataSet) {
				int prediction = predict(inputs, weights);
				double error = inputs[inputs.length - 1] - prediction;
				sumError += Math.pow(error, 2);
				// Update the bias
				weights[0] = weights[0] + learningRate * error;
				// Update each weight
				for (int j = 0; j < inputs.length - 1; j++) {
					weights[j+1] += learningRate *error * inputs[j];
				}
			}
			if (sumError == 0) {
				break;
			}
			System.out.printf("Epoch: %d | errorSum: %.2f | ", i+1, sumError);	
			printWeights(weights);
			System.out.printf(" | Accuracy: %.2f%%\n", accuracy(weights, dataSet));
		}
		return weights;
	}
	
	public static double[] updatePocketWeights(ArrayList<double[]> dataSet, double learningRate, int numOfEpochs) {
		// Initialize the weight and pocketWeight vector to be all zeros
		double[] weights = new double[dataSet.get(0).length];
		double[] pocketWeights = new double[dataSet.get(0).length];
		for (int i = 0; i < dataSet.get(0).length - 1; i++) {
			weights[i] = 0;
			pocketWeights[i] = 0;
		}
		
		// Calculate the summation of the errors for each set of inputs
		double pocketErrorSum = 0;
		for (int i = 0; i < numOfEpochs; i++) {
			double sumError = 0;
			for (double[] inputs : dataSet) {
				int prediction = predict(inputs, weights);
				double error = inputs[inputs.length - 1] - prediction;
				sumError += Math.pow(error, 2);
				// Update the bias
				weights[0] = weights[0] + learningRate * error;
				// Update each weight
				for (int j = 0; j < inputs.length - 1; j++) {
					weights[j+1] += learningRate *error * inputs[j];
				}
			}
			if (sumError == 0) {
				break;
			}
			if (accuracy(weights, dataSet) > accuracy(pocketWeights, dataSet)) {
				pocketWeights = Arrays.copyOf(weights, weights.length);
				pocketErrorSum = sumError;
				
			}
			System.out.printf("Epoch: %d | errorSum: %.2f | ", i+1, pocketErrorSum);	
			printWeights(pocketWeights);
			System.out.printf(" | Accuracy: %.2f%%\n", accuracy(pocketWeights, dataSet));
		}
		return pocketWeights;
	}
	
	public static void printWeights(double[] weights) {
		System.out.print("Weights: ");
		for (double weight : weights) {
			if (weight == weights[weights.length -1]) {
				System.out.printf("%.2f", weight);
			}
			else {
				System.out.printf("%.2f, ", weight);
			}
		}
	}
	
	public static ArrayList<double[]> parseFileData() throws Exception {
		// Get Data from CSV file
		URL path = ClassLoader.getSystemResource("Breast_cancer_data.csv");
		File file = new File(path.toURI());
		BufferedReader reader = new BufferedReader(new FileReader(file));
		
		ArrayList<double[]> dataSet = new ArrayList<double[]>();
		
		boolean firstLine = true;
		String line;
		while((line = reader.readLine()) != null) {
			// Don't need the first line for calculations
			if (firstLine) {
				firstLine = false;
			}
			else {
				String[] stringValues = line.split(",");
				double[] values = new double[stringValues.length];
				// Convert each value in the line to a double
				for (int i = 0; i < stringValues.length; i++) {
					values[i] = Double.parseDouble(stringValues[i]);
				}
				// Add the array of doubles to the data set
				dataSet.add(values);
			}
		}
		return dataSet;	
	}
	
	public static double accuracy(double[] weights, ArrayList<double[]> dataSet) {
		int counter = 0;
		double total = dataSet.size();
		for (double[] inputs: dataSet) {
			if (predict(inputs, weights) == inputs[inputs.length -1]) {
				counter++;
			}
		}
		return (counter/total) * 100;
	}
	
	public static void main(String[] args) {
		ArrayList<double[]> dataSet = new ArrayList<double[]>();
		try {
			dataSet = parseFileData();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		
		double learningRate = .1;
		int epochs = 100;
		
		Scanner scanner = new Scanner(System.in);
		boolean validInput = false;
		while (!validInput) {
			System.out.print("Enter 1 for naive perceptron or 2 for pocket perceptron: ");
			String input = scanner.nextLine();
			if (input.equals("1")) {
				validInput = true;
				double[] weights = updateWeights(dataSet, learningRate, epochs);
				System.out.print("Final ");
				printWeights(weights);
				System.out.printf(" | Accuracy: %.2f%%\n", accuracy(weights, dataSet));
			}
			else if (input.equals("2")) {
				validInput = true;
				double[] pocketWeights = updatePocketWeights(dataSet, learningRate, epochs);
				System.out.print("Final ");
				printWeights(pocketWeights);
				System.out.printf(" | Accuracy: %.2f%%\n", accuracy(pocketWeights, dataSet));
			}
		}
		scanner.close();
	}
}

