package ai.certifai.solution;

package global.skymind.question2;

import org.datavec.image.transform.*;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.yaml.snakeyaml.reader.ReaderException;
import org.slf4j.Logger;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/* ===================================================================
 * We will solve a task of classifying horse breeds.
 * The dataset contains 4 classes, each with just over 100 images
 * Images are of 256x256 RGB
 *
 * Source: https://www.kaggle.com/olgabelitskaya/horse-breeds
 * ===================================================================


 */
public class HorseBreedClassifier {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(HorseBreedClassifier.class);
    private static final int height = 64;
    private static final int width = 64;
    private static final int nChannel = 3;
    private static final int nOutput = 4;
    private static final int seed = 141;
    private static Random rng = new Random(seed);
    private static double lr = 1e-4;
    private static final int nEpoch = 20;
    private static final int batchSize = 3;




    public static void main(String[] args) throws IOException {

        HorseBreedIterator.setup();

        //method all static which means method that have static method belng to that class not the instance
        HorseBreedIterator iterator = new HorseBreedIterator();


        //Image Transformation

        ImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform cropImage = new CropImageTransform(25);
        ImageTransform rotateImage = new RotateImageTransform(rng, 15);
        ImageTransform showImage = new ShowImageTransform("Image",1000);
        boolean shuffle = false;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip,0.5),
                new Pair<>(rotateImage, 0.5),
                new Pair<>(cropImage,0.3),
                new Pair<>(showImage,1.0)
        );
        ImageTransform transform = new PipelineImageTransform();

        DataSetIterator trainIter = HorseBreedIterator.getTrain(transform,batchSize);
        DataSetIterator testIter = HorseBreedIterator.getTest(1);

        //model configuration

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .activation(Activation.RELU)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .convolutionMode(ConvolutionMode.Same)
                .list()
                .layer(0, new ConvolutionLayer.Builder()
                        .kernel (3,3)
                        .stride(1,1)
                        .nIn(nChannel)
                        .nOut(12) //whatever value //num of filter and depth sise
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder()
                        .kernelSize(3,3)
                        .stride(2,2)
                        .padding(1,1)
                        .poolingType(Subsampling.PoolingType.MAX)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(50)
                        .build())
                .layer(3, new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKEHOOD) //Multiclass Classification
                        .nOut(nOutput)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, nChannel))
                .build();


        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        log.info("**************************************** MODEL SUMMARY ****************************************");
        System.out.println(model.summary());

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        model.setListener(new ScoreIterartionListener(10));
        model.fit(trainIter, nEpoch);


        log.info("**************************************** MODEL EVALUATION ****************************************");

        // Perform evaluation on both train and test set
        Evaluation evalTrain = model.evaluate(trainIter);
        Evauation evalTest = model.evaluate(testIter);

        System.out.println("Train Evaluate : "+ evalTrain.stats());
        model.evaluate(trainIter);
        System.out.println("Test Evaluate : "+ evalTest.stats());
        model.evaluate(testIter);


}