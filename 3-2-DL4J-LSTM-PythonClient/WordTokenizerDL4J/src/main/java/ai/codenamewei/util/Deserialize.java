package ai.codenamewei.util;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.TypeFactory;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class Deserialize
{
    public Map<String, String> deserialize(String filePath)
    {
        Map<String, String> labelClass = new HashMap<>();
        try {
            // create object mapper instance
            ObjectMapper mapper = new ObjectMapper();

            // convert JSON file to map
            labelClass = mapper.readValue(new File(filePath), Map.class);

            System.out.println("Deserialization: " + labelClass);

        } catch (Exception ex) {
            ex.printStackTrace();
        }


        return labelClass;
    }
}
