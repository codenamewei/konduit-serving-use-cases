package ai.codenamewei;

import ai.codenamewei.util.Deserialize;
import ai.codenamewei.util.Serialize;
import org.deeplearning4j.iterator.BertIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class LocalInference
{
    public static void writeLabelJson(String dataPath, String labelJson)
    {
        Map<String, String> labelClass = new HashMap<>();
        Integer labelIndex = 0;

        File fileDataPath = new File(dataPath);

        File[] subFileDataPath = fileDataPath.listFiles();

        for (File file : subFileDataPath)
        {
            if (file.isDirectory())
            {
                labelClass.put(file.getName(), Integer.toString(labelIndex));
                ++labelIndex;
            }
        }

        new Serialize().serialize(labelClass, labelJson);
    }


    public static void main(String args[]) throws IOException
    {
        String root_path = "C:\\Users\\chiaw\\Documents\\data\\konduit-serving-use-cases\\3-2-DL4J-LSTM-PythonClient\\";
        String path = root_path + "bert-base-uncased-vocab.txt";
        String rootDataPath = "C:\\Users\\chiaw\\Documents\\data\\20news-bydate\\20news-bydate-test\\";
        String sampleInputPath = rootDataPath + "alt.atheism\\53068";
        String sampleLabel = "default";
        String modelPath = root_path + "bert.zip";

        String labelJsonPath = "labelClass.json";

        //writeLabelJson(rootDataPath, root_path + labelJsonPath);

        Map<Integer, String> labelClass = new Deserialize().deserialize(root_path + labelJsonPath);

        File vocabFile = new File(path);
        BertWordPieceTokenizerFactory tokenizer;

        try {

            tokenizer = new BertWordPieceTokenizerFactory(vocabFile, true, true, StandardCharsets.UTF_8);

        } catch (IOException e) {
            throw new IllegalStateException("Vocabulary file missing");
        }

        //List<String> sentencesArr = new ArrayList<>();
        //sentencesArr.add(inputSentence);
        //LabeledSentenceProvider provider = new CollectionLabeledSentenceProvider(sentencesArr, labelsArr, new Random(123));

        Map filesByLabel = new HashMap<String, List<File>>();

        List dataList = new ArrayList<File>();
        dataList.add(new File(sampleInputPath));

        filesByLabel.put(sampleLabel, dataList);


        BertIterator bertIterator = BertIterator.builder()
                .tokenizer(tokenizer)
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 256)
                .minibatchSize(1)
                .sentenceProvider(new FileLabeledSentenceProvider(filesByLabel))
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK)
                .vocabMap(tokenizer.getVocab())
                .task(BertIterator.Task.SEQ_CLASSIFICATION)
                .build();


        File savedGraphPath = new File(modelPath);
        ComputationGraph model = ModelSerializer.restoreComputationGraph(savedGraphPath.getAbsolutePath());

        if(bertIterator.hasNext())
        {
            //Get Local Inference
            INDArray[] classes = model.output(bertIterator);

            //System.out.println(classes[0].toString());

            INDArray classIndex = Nd4j.argMax(classes[0], 1);

            System.out.println("Output Class: " + classIndex);

            System.out.println("Output Label: " + labelClass.get((int)classIndex.getDouble(0, 0)));

            bertIterator.reset();

            //Save as feature and mask
            MultiDataSet dataset = bertIterator.next();

            INDArray feature = dataset.getFeatures()[0];
            INDArray featureMask = dataset.getFeaturesMaskArrays()[0];

            File featureFile = new File(root_path + "feature.npy");
            File featureMaskFile = new File(root_path + "featureMask.npy");

            Nd4j.writeAsNumpy(feature, featureFile);
            Nd4j.writeAsNumpy(featureMask, featureMaskFile);
        }

    }
}
