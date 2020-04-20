package ai.codenamewei.util;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.TypeFactory;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class Deserialize
{
    public Map<Integer, String> deserialize(String filePath)
    {
        Map<String, String> labelClassBuffer = new HashMap<>();
        try {
            // create object mapper instance
            ObjectMapper mapper = new ObjectMapper();

            // convert JSON file to map
            labelClassBuffer = mapper.readValue(new File(filePath), Map.class);

            //System.out.println("Deserialization: " + labelClassBuffer);

        } catch (Exception ex) {
            ex.printStackTrace();
        }

        Map<Integer, String> classLabel = new HashMap<>();

        for (Map.Entry<String,String> entry : labelClassBuffer.entrySet())
        {
            classLabel.put(Integer.parseInt(entry.getValue()), entry.getKey());
        }
        return classLabel;
    }
}
