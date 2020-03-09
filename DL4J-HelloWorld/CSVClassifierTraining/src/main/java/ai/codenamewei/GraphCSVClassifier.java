package ai.codenamewei;

import lombok.extern.log4j.Log4j;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

@Log4j
public class GraphCSVClassifier
{
    public static void main(String[] args) throws Exception
    {
        int seed = 123;
        double learningRate = 0.01;
        int batchSize = 50;
        int nEpochs = 10;

        int numInputs = 2;
        int numClasses = 2;
        int numHiddenNodes = 20;

        ClassLoader classLoader = ClassLoader.getSystemClassLoader();
        File trainFile = new File(classLoader.getResource("linear_data_train.csv").getFile());

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(trainFile));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,2);

        ComputationGraphConfiguration graphConfig = new NeuralNetConfiguration.Builder()
                .updater(new Nesterovs(learningRate, 0.9))
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("input")
                .addLayer("hiddenlayer1", new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build(), "input")
                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numClasses).build(), "hiddenlayer1")
                .setOutputs("output")
                .build();


        ComputationGraph model = new ComputationGraph(graphConfig);
        model.init();

        model.setListeners(new ScoreIterationListener(10));

        model.fit(trainIter, nEpochs );

        File testFile = new File(classLoader.getResource("linear_data_eval.csv").getFile());
        rr.initialize(new FileSplit(testFile));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rr, batchSize,0,numClasses);

        Evaluation eval = model.evaluate(testIter);

        System.out.println(eval.stats());

        File modelSavedPath = new File(System.getProperty("java.io.tmpdir"), "dl4j_csv_graph.zip");
        ModelSerializer.writeModel(model, modelSavedPath, false);

       log.info("Model saved at " +modelSavedPath.toString());

       log.info("Program end.");

    }

}
