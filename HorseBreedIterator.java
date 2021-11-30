package global.skymind.question2;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;


import java.io.File;
import java.io.IOException;
import java.util.Random;


public class HorseBreedIterator {

    private static final String[] allowedExt = BaseImageLoader.ALLOWED_FORMATS;
    private static final int height = 64;
    private static final int width = 64;
    private static final int nChannel = 3;
    private static final int numLabels = 4;
    private static final int seed = 141;
    private static int batchSize;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static Random rng = new Random(seed);
    private static InputSplit trainData, testData;
    private static ImageTransform imageTransform;
    private static DataNormalization scaler = new ImagePreProcessingScaler(0,1);

    public static void setup() throws IOException {

        File input = new ClassPathResource("HorseBreed").getFile();


        FileSplit filesInDir = new FileSplit(input);

        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        PathFilter bPF = new BalancePathFilter (rng, allowedExt, labelMaker);


        InputSplit[] allData = filesInDir.sample(pathFilter, 80, 20);
        trainData = allData[0];
        testData = allData[1];
    }

    private static DataSetIterator makeIterator(boolean train) throws IOException { //private so tak boleh guna in main tapi dalam one sheet ni boleh

        ImageRecordReader rr = new ImageRecordReader(height,width,nChannel,labelMaker);


        if (train && imageTransform != null){
            rr.initialize(trainData,imageTransform);
        }else{
            rr.initialize(testData);
        }
        DataSetIterator iter = new RecordReaderDataSetIterator(rr, batchSize, 1, numLabels);
        iter.setPreProcessor(scaler);
        return iter;
    }

    public static DataSetIterator getTrain(ImageTransform transform, int batchsize) throws IOException {
        imageTransform = transform;
        batchSize = batchsize;
        return makeIterator(true);
    }

    public static DataSetIterator getTest(int batchsize) throws IOException {
        batchSize = batchsize;
        return makeIterator(false);
    }
}