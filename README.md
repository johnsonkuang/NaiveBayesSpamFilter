# CSE 312: Naive Bayes

**IMPORTANT**: Try to get started on Naive Bayes early so that you can come to the Office Hours in case you face problems with any of the parts.

## 0\. Outline

1. [Setting up Git](README.md#1-getting-started)
2. [Information about the files and file structure](README.md#2-coding-it-up)
3. [Submitting your assignment](README.md#3-submitting-your-assignment)

## 1\. Getting started

The following instructions are written for a Unix-based platform (e.g., Linux, MacOS, etc.).
For Windows, you have to download [Git for Windows](https://gitforwindows.org/) and then you can follow the instructions below.

### 1.1\. Working with Git

We will be using `git`, a source code control tool. This will allow you to download the code for the assignment, and also submit it in a standardized format that will streamline grading.

You will also be able to use `git` to commit your progress on the labs
as you go. This is **important**: Use `git` to back up your work. Back
up regulary by both committing and pushing your code as we describe below.

Course git repositories will be hosted as a repository in [GitLab](https://gitlab.cs.washington.edu). Your code will be in a private repository that is visible only to you and the course staff.

#### 1.1.1\. Getting started with Git

There are numerous guides on using `git` that are available. They range from being interactive to just text-based. Find one that works and experiment -- making mistakes and fixing them is a great way to learn. Here is a [link to resources](https://help.github.com/articles/what-are-other-good-resources-for-learning-git-and-github) that GitHub suggests starting with. If you have no experience with `git`, you may find this [web-based tutorial helpful](https://try.github.io/levels/1/challenges/1).

Git may already be installed in your environment; if it's not, you'll need to install it first. For `bash`/Linux environments, git should be a simple `apt-get` / `yum` / etc. install. More detailed instructions may be [found here](http://git-scm.com/book/en/Getting-Started-Installing-Git).

If you are using Eclipse or IntelliJ, many versions come with git already configured. The instructions will be slightly different than the command line instructions listed but will work for any OS. For Eclipse, detailed instructions can be found at [EGit User Guide](http://wiki.eclipse.org/EGit/User_Guide) or [EGit Tutorial](http://eclipsesource.com/blogs/tutorials/egit-tutorial).

#### 1.1.2\. Cloning your Naive Bayes repository

We've created a GitLab repository that you will use to implement Naive Bayes. This repository is hosted on the [CSE GitLab](https://gitlab.cs.washington.edu) site, and you can view it by visiting the GitLab website at `https://gitlab.cs.washington.edu/cse312-20sp/naive-bayes-[your GitLab username]`. If you don't see this repository or are unable to access it, let us know immediately!

The first thing you'll need to do is set up a SSH key to allow communication with GitLab:

1. If you don't already have one, generate a new SSH key. See [these instructions](http://doc.gitlab.com/ce/ssh/README.html) for details on how to do this.
2. Visit the [GitLab SSH key management page](https://gitlab.cs.washington.edu/profile/keys). You'll need to log in using your CSE account.
3. Click "Add SSH Key" and paste in your **public** key into the text area.

While you're logged into the GitLab website, browse around to see which projects you have access to. You should have access to `naive-bayes-[your username]`.

We next want to move the code from the GitLab repository onto your local file system. To do this, you'll need to clone the lab repository by issuing the following commands on the command line:

```sh
$ git clone git@gitlab.cs.washington.edu:cse312-20sp/naive-bayes-MY_GITLAB_USERNAME.git
$ cd naive-bayes-MY_GITLAB_USERNAME
```

This will make a complete replica of the assignment repository locally. If you get an error that looks like:

```sh
Cloning into 'naive-bayes-myusername'...
Permission denied (publickey).
fatal: Could not read from remote repository.
```

... then there is a problem with your GitLab configuration. Check to make sure that your GitLab username matches the repository suffix, that your private key is in your SSH directory (`~/.ssh`) and has the correct permissions, and that you can view the repository through the website.

Cloning will make a complete replica of the lab repository locally. Any time you `commit` and `push` your local changes, they will appear in the GitLab repository.  Since we'll be grading the copy in the GitLab repository, it's important that you remember to push all of your changes!

Let's test out the repository by doing a push of your master branch to GitLab. Do this by issuing the following commands:

```sh
$ touch empty_file
$ git add empty_file
$ git commit -a -m 'Testing git'
$ git push # ... to origin by default
```

The `git push` tells git to push all of your **committed** changes to Gitlab.

After executing these commands, you should see something like the following:

```sh
Counting objects: 4, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (2/2), done.
Writing objects: 100% (3/3), 286 bytes | 0 bytes/s, done.
Total 3 (delta 1), reused 0 (delta 0)
To git@gitlab.cs.washington.edu:cse312-20sp/naive-bayes-username.git
   cb5be61..9bbce8d  master -> master
```

We pushed a blank file to our directory, which isn't very interesting. Let's clean up after ourselves:

```sh
$ # Tell git we want to remove this file from our repository
$ git rm empty_file
$ # Now commit all pending changes (-a) with the specified message (-m)
$ git commit -a -m 'Removed test file'
$ # Now, push this change to GitLab
$ git push
```

If you don't know Git that well, this probably seemed very arcane. Just keep using Git and you'll understand more and more. We'll provide explicit instructions below on how to use these commands to actually indicate your final lab solution.

## 2\. Coding it up
**IMPORTANT**: Please go to [resource tab on Ed Discussion board](https://us.edstem.org/courses/377/resources) to download the naivebayesnote.pdf and naivebayesslides.pdf. They are very helpful supplementary matrials for this homework and the note includes a very detailed walkthrough of the maths.
### 2.1\. Starter code

#### 2.1.1\. Data

Inside the `data` folder, the emails
are separated into `train` and `test` data. Each
`train` email is already labeled as either *spam*
or *ham*, and they should be used to train your model
and word probabilities. The `test` data is not labeled,
and they are the emails whose labels you will predict using
your classifier.

The emails we are using are a subset of the Enron Corpus,
which is a set of real emails from employees at an energy
company. The emails have a subject line and a body, both of
which are `tokenized` so that each unique word or bit of
punctuation is separated by a space or newline. The starter
code provides a function that takes a filename and returns a
set of all of the distinct tokens in the file.

**Do not alter any of the data files to ensure accurate results.**

#### 2.1.2\. SpamFilterMain.java

The provided main executable file that handles file loading
for you and calls the NaiveBayes that you'll implement. **DO NOT MODIFY THIS FILE**.

You will not turn in this file. The only file you will turn in
is `NaiveBayes.java`, which is expected to be run with the given version of `SpamFilterMain.java`.

#### 2.1.3\. NaiveBayes.java

`NaiveBayes.java`: The file you will modify and implement. A few notes:

* **Do NOT modify the provided method headers**.
    Again, the `NaiveBayes.java` you turn in is expected to be run with the given `SpamFilterMain.java`.
* Think about the data structures you want to use to keep track of the word counts and/or probabilities.

### 2.2\. Running the program

To run the program, first compile it with:

```sh
$ javac SpamFilterMain.java
```

then, execute it with:

```sh
$ java SpamFilterMain
```

Note, the `data` directory needs to be in the same
directory in which the program is executed. If you are running into issues loading the data (especially if you are using Eclipse) and you're not sure where to put the `data` directory, check the console output produced when you run `SpamFilterMain`. The console output prints out the current working directory (cwd) where the program 
is executing. Just move the entire `data` directory into that cwd that was printed.

### 2.3\. Comparing the result:

It is not expected that Naive Bayes will classify every single
test email correctly, but it should certainly do better than random chance! We are not grading you on whether you classify 100% of the examples accurately, but rather on general program correctness.

After you've classified the 500 test emails, you can compare your results with the actual labels that we hid from you, by using the output checker [here](https://courses.cs.washington.edu/courses/cse312/19wi/nbc/checker.html).

For this specific test dataset, you should get an error score of **27** (total number of incorrectly classified emails). Note that we will run your code on a test dataset you haven't seen.

### 2.4\. Where to look if you get wrong error score

* Only defining P(w | spam) for words in the spam
training data rather than in both spam and ham.  Similarly for
P(w | ham).
* In testing, trying to use words that didn't appear
anywhere in the training data.
* In testing, only using words that appeared in both spam
and ham training data.
* Laplace smoothing using constants other than +1 in
numerator and +2 in denominator.
* Dividing numbers of type integer.
* Forgetting to use log probabilities to prevent underflow
as discussed in the notes.
* Multiplying log probabilities together as in Bayes'
Theorem and comparing to 0.5 instead of adding log
probabilities and comparing them as explained in the notes.

### 2.5\. Notes and advice

* Read about how to avoid floating point underflow in the notes.
* Make sure you understand how smoothing works.
* **Remember to remove any debug statements that you are
    printing to the output.**
* If you use Eclipse, remove all package statements before you turn in your source code.
* Needless to say, you should practice what you've learned
    in other courses:  
  * document your program, use good variable names, keep your code clean and straightforward, etc.
  * Include comments outlining what your program does and how. We will not spend time trying to decipher obscure, contorted code.

## 3\. Submitting your assignment

You may submit your code multiple times; we will use the latest version you submit that arrives before the deadline.

The criteria for your lab being submitted on time is that your code must be pushed by the due date and time. This means that if one of the TAs or the instructor were to open up GitLab, they would be able to see your solutions on the GitLab web page.

## Using Git for turn-in

**Make sure that you are in the same directory as NaiveBayes.java**

```sh
$ # Now commit all pending changes (-a) with the specified message (-m)
$ git commit -a -m '[insert commit message here, it can be anything]'
$ # Now, push this change to GitLab
$ git push
```

Just because your code has been committed on your local machine does not mean that it has been submitted -- it needs to be on GitLab!

**Confirm that you see the correct file on GitLab.**
**If you make any changes on GitLab after the deadline you will get the late penalty.**

#### Final Word of Caution for Git!

Git is a distributed version control system. This means everything operates offline until you run `git pull` or `git push`. This is a great feature.

The bad thing is that you may **forget to `git push` your changes**. This is why we strongly, strongly suggest that you **check GitLab to be sure that what you want us to see matches up with what you expect**.  As a second sanity check, you can re-clone your repository in a different directory to confirm the changes:

```sh
$ git clone git@gitlab.cs.washington.edu:cse312-20sp/naive-bayes-username.git confirmation_directory
$ cd confirmation_directory
$ # ... make sure everything is as you expect ...
```

We've had a lot of fun designing this assignment, and we hope you enjoy hacking on it!