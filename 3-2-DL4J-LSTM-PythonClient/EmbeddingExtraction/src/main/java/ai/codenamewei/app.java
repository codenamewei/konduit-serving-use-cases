package ai.codenamewei;

import org.datavec.api.writable.NDArrayWritable;
import org.deeplearning4j.iterator.BertIterator;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class app
{
    private static String pathToVocab = "C:\\Users\\chiaw\\Documents\\data\\konduit-serving-use-cases\\3-2-DL4J-LSTM-PythonClient\\bert-base-uncased-vocab.txt";
    private static String textSampleFile = "C:\\Users\\chiaw\\Documents\\data\\konduit-serving-use-cases\\3-2-DL4J-LSTM-PythonClient\\alt-atheism-sample.txt";


    public static Map<String, List<File>> getFiles()
    {
        Map filesByLabel = new HashMap<String, List<File>>();

        List listFile = new ArrayList<File>();
        listFile.add(new File(textSampleFile));

        filesByLabel.put("default", listFile);

        return filesByLabel;
    }

    public static void main(String[] args) throws Exception
    {
        //Key: label. Value: list of files for that label
        Map filesByLabel = getFiles();

        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(new File(pathToVocab), true, true, StandardCharsets.UTF_8);

        final Integer MAXLENGTH = 256;

        BertIterator b = BertIterator.builder()
                .tokenizer(t)
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, MAXLENGTH)
                .minibatchSize(1)
                .sentenceProvider(new FileLabeledSentenceProvider(filesByLabel))
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK)
                .vocabMap(t.getVocab())
                .task(BertIterator.Task.SEQ_CLASSIFICATION)
                .build();


        while(b.hasNext())
        {
            MultiDataSet dataset = b.next();
            INDArray feature = dataset.getFeatures()[0];
            INDArray featureMask = dataset.getFeaturesMaskArrays()[0];

            File featureFile = new File("feature.npy");
            File featureMaskFile = new File("featureMask.npy");

            Nd4j.saveBinary(feature, featureFile);
            Nd4j.saveBinary(featureMask, featureMaskFile);

            INDArray featureRecovered = Nd4j.readBinary(featureFile);
            INDArray featureMaskRecovered = Nd4j.readBinary(featureMaskFile);

        }

    }
}
