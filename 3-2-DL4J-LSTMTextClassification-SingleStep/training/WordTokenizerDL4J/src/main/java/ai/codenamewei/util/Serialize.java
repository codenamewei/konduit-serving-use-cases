package ai.codenamewei.util;

import com.fasterxml.jackson.core.JsonGenerationException;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;

import java.io.File;
import java.io.IOException;
import java.util.Map;

public class Serialize
{
    public void serialize(Map<String, String> input, String savedPath)
    {
        try {

            ObjectMapper mapper = new ObjectMapper();

            mapper.writeValue(new File(savedPath), input);

            // convert map to JSON string
            String json = mapper.writeValueAsString(input);
            System.out.println("Serialization: " + json);   // compact-print


        } catch (JsonProcessingException e)
        {
            e.printStackTrace();

        } catch (IOException e)
        {
            e.printStackTrace();
        }

    }
}
