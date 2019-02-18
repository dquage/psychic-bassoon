package dl;

import org.datavec.api.records.converter.RecordReaderConverter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.hadoop.records.reader.mapfile.MapFileRecordReader;
import org.datavec.hadoop.records.writer.mapfile.MapFileRecordWriter;

import java.io.File;
import java.util.Random;

public class MapFileConversion {

    public static void main(String[] args) throws Exception {

        //Create CSV reader
        File irisFile = new ClassPathResource("iris.txt").getFile();
        RecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(new FileSplit(irisFile));

        //Create map file writer
        File mapFileLoc = new File("mapFileOutputDir");
        MapFileRecordWriter writer = new MapFileRecordWriter(mapFileLoc);

        //Convert to MapFile binary format:
        RecordReaderConverter.convert(recordReader, writer);


        //Read back in from binary MapFile, random order:
        Random rng = new Random(12345);
        RecordReader mapFileReader = new MapFileRecordReader(rng);
        mapFileReader.initialize(new FileSplit(mapFileLoc));

        //Print out:
        while(mapFileReader.hasNext()){
            System.out.println(mapFileReader.next());
        }

    }
}
