package org.aksw.gerbil.ws4test;

import java.io.IOException;
import java.io.InputStream;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

import org.aksw.gerbil.transfer.nif.Document;
import org.aksw.gerbil.transfer.nif.Marking;
import org.aksw.gerbil.transfer.nif.Span;
import org.aksw.gerbil.transfer.nif.TurtleNIFDocumentCreator;
import org.aksw.gerbil.transfer.nif.TurtleNIFDocumentParser;
import org.aksw.gerbil.transfer.nif.data.NamedEntity;
import org.restlet.representation.Representation;
import org.restlet.resource.Post;
import org.restlet.resource.ServerResource;
import org.apache.commons.io.IOUtils;
import org.apache.http.HttpEntity;
import org.apache.http.StatusLine;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;



public class LocalIntermediateWebserver extends ServerResource {

    private TurtleNIFDocumentParser parser = new TurtleNIFDocumentParser();
    private TurtleNIFDocumentCreator creator = new TurtleNIFDocumentCreator();

    private String clusterServiceURL = "http://localhost:5555/cluster_ed";
    
    private Gson gson = new GsonBuilder().create();
    private HttpClient client = HttpClients.createDefault();    
    
    @Post
    public String accept(Representation request) {
    	System.out.println("-------------------------------------------------");
        Reader inputReader;
        try {
            inputReader = request.getReader();
        } catch (IOException e) {
            System.err.println("Exception while reading request." + e.getMessage());
            return "";
        }
        Document document;
        try {
            document = parser.getDocumentFromNIFReader(inputReader);
        } catch (Exception e) {
        	System.err.println("Exception while reading request." + e.getMessage());
            return "";
        }

        List<Marking> markings = document.getMarkings();
        String text = document.getText();                           

        //System.out.println("num spots = " + markings.size());
        List<Marking> entities = sendRequestToCluster(text, markings);  
        //System.out.println("   Solution size = " + entities.size());         
        
        // ... this new list is added to the document and the document is
        // send back to GERBIL
        document.setMarkings(entities);
        String nifDocument = creator.getDocumentAsNIFString(document);
        return nifDocument;
    }
    
    
    public List<Marking> sendRequestToCluster(String text , List<Marking> markings) {
    	List<Marking> entities = new ArrayList<Marking>(markings.size());
    	
    	JsonObject out = null;
    	try {
			out = queryJson(text, markings, clusterServiceURL);
		} catch (IOException e) {
			e.printStackTrace();
		}
    	
    	// process the response
    	/*if (out!= null) {
    		JsonArray outMentionsJson = out.getAsJsonArray("annotations");
    		for (JsonElement je : outMentionsJson) {
    			JsonObject mentionJson = je.getAsJsonObject();
    			int start = mentionJson.get("start").getAsInt();
    			int length = mentionJson.get("length").getAsInt();
    			String entity = mentionJson.get("entity").getAsString();
    			entities.add(new NamedEntity(start, length, entity));        		
    		}
    	}*/
    	
    	return entities;
    }
    
    
    private JsonObject queryJson(String text, List<Marking> markings, String url) throws IOException {

        JsonObject parameters = new JsonObject();

        if (markings != null) {
            JsonArray mentionsJson = new JsonArray();
            for (Marking m : markings) {
            	Span sp = (Span) m;
                JsonObject mentionJson = new JsonObject();
                mentionJson.addProperty("start", sp.getStartPosition());
                mentionJson.addProperty("length", sp.getLength());
                mentionsJson.add(mentionJson);
            }
            parameters.add("spans", mentionsJson);
        }
        parameters.addProperty("text", text);

        HttpPost request = new HttpPost(url);
        request.addHeader("Content-Type", "application/json");
        request.setEntity(new StringEntity(gson.toJson(parameters), "UTF8"));
        request.addHeader("Accept", "application/json");

        CloseableHttpResponse response = (CloseableHttpResponse) client.execute(request);
        InputStream is = null;
        HttpEntity entity = null;
        try {
            StatusLine status = response.getStatusLine();
            if ((status.getStatusCode() < 200) || (status.getStatusCode() >= 300)) {
                entity = response.getEntity();
                System.err.println("The response had a wrong status: \"" + status.toString() + "\". Content of response: \""
                        + IOUtils.toString(entity.getContent()) + "\". Returning null.");
                return null;
            }
            entity = response.getEntity();
            is = entity.getContent();
            return new JsonParser().parse(IOUtils.toString(is)).getAsJsonObject();
        } catch (Exception e) {
        	System.err.println("Couldn't request annotation for given text. Returning null." + e.getMessage());
        } finally {
            IOUtils.closeQuietly(is);
            if (entity != null) {
                EntityUtils.consume(entity);
            }
            if (response != null) {
                response.close();
            }
        }
        return null;
    }    
}
