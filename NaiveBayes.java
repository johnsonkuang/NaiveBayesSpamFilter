// Do not submit with package statements if you are using eclipse.
// Only use what is provided in the standard libraries.

import java.io.*;
import java.util.*;

public class NaiveBayes {

    private Map<String, Double> probabilitySpamMap;
    private Map<String, Double> probabilityHamMap;

    private double probSpam;
    private double probHam;

    private double numSpam;
    private double numHam;

    public NaiveBayes(){
        probabilitySpamMap = new HashMap<>();
        probabilityHamMap = new HashMap<>();

        probSpam = 0.0;
        probHam = 0.0;
    }

    /*
     * !! DO NOT CHANGE METHOD HEADER !!
     * If you change the method header here, our grading script won't
     * work and you will lose points!
     * 
     * Train your Naive Bayes Classifier based on the given training
     * ham and spam emails.
     *
     * Params:
     *      hams - email files labeled as 'ham'
     *      spams - email files labeled as 'spam'
     */
    public void train(File[] hams, File[] spams) throws IOException {
        propagateProbabilities(hams, probabilityHamMap);
        propagateProbabilities(spams, probabilitySpamMap);

        probHam = (double) hams.length / (hams.length + spams.length);
        probSpam = (double) spams.length / (hams.length + spams.length);

        numSpam = spams.length;
        numHam = hams.length;
    }

    private void propagateProbabilities(File[] emails, Map<String, Double> map) throws IOException{
        Map<String, Integer> wordCount = new HashMap<>();
        for(File email : emails){
            Set<String> wordList = tokenSet(email);
            //counts each word once because it's a set
            for(String word : wordList){
                if(!wordCount.containsKey(word)){
                    wordCount.put(word, 0);
                }
                wordCount.put(word, wordCount.get(word) + 1);
            }
        }

        for(String word: wordCount.keySet()){
            map.put(word, (double) ((wordCount.get(word) + 1) / (emails.length + 2)));
        }
    }

    /*
     * !! DO NOT CHANGE METHOD HEADER !!
     * If you change the method header here, our grading script won't
     * work and you will lose points!
     *
     * Classify the given unlabeled set of emails. Add each email to the correct
     * label set. SpamFilterMain.java would follow the format in
     * example_output.txt and output your result to stdout. Note the order
     * of the emails in the output does NOT matter.
     * 
     *
     * Params:
     *      emails - unlabeled email files to be classified
     *      spams  - set for spam emails that needs to be populated
     *      hams   - set for ham emails that needs to be populated
     */
    public void classify(File[] emails, Set<File> spams, Set<File> hams) throws IOException {
        for(File email: emails){
            Set<String> words = tokenSet(email);

            double emailProbSpam = 0.0;
            double emailProbHam = 0.0;

            for(String word: words){
                if(probabilitySpamMap.containsKey(word)){
                    emailProbSpam += Math.log(probabilitySpamMap.get(word));
                } else {
                    emailProbSpam += Math.log((double) 1 / (numSpam + 2));
                }

                if(probabilityHamMap.containsKey(word)){
                    emailProbHam += Math.log(probabilityHamMap.get(word));
                } else {
                    emailProbHam += Math.log((double) 1 / (numHam + 2));
                }
            }

            emailProbSpam += Math.log(probSpam);
            emailProbHam += Math.log(probHam);

            if(emailProbSpam > emailProbHam){
                spams.add(email);
            } else {
                hams.add(email);
            }
        }
    }


    /*
     *  Helper Function:
     *  This function reads in a file and returns a set of all the tokens. 
     *  It ignores "Subject:" in the subject line.
     *  
     *  If the email had the following content:
     *  
     *  Subject: Get rid of your student loans
     *  Hi there ,
     *  If you work for us , we will give you money
     *  to repay your student loans . You will be 
     *  debt free !
     *  FakePerson_22393
     *  
     *  This function would return to you
     *  ['be', 'student', 'for', 'your', 'rid', 'we', 'of', 'free', 'you', 
     *   'us', 'Hi', 'give', '!', 'repay', 'will', 'loans', 'work', 
     *   'FakePerson_22393', ',', '.', 'money', 'Get', 'there', 'to', 'If', 
     *   'debt', 'You']
     */
    public static HashSet<String> tokenSet(File filename) throws IOException {
        HashSet<String> tokens = new HashSet<String>();
        Scanner filescan = new Scanner(filename);
        filescan.next(); // Ignoring "Subject"
        while(filescan.hasNextLine() && filescan.hasNext()) {
            tokens.add(filescan.next());
        }
        filescan.close();
        return tokens;
    }
}
