# -*- coding: utf-8 -*-
from textblob import TextBlob
import os
import re
import datetime
from os import listdir
from os.path import isfile, join
import cPickle as pickle

from textblob.classifiers import NaiveBayesClassifier

_path = "/home/pyaria/Projects/ai-hackathon/bad-news-sell"

print "loading pickle"
pickle_file_name = "realfake.pickle"
# with open(pickle_file_name, 'rb') as pickle_file:
# 	cl = pickle.load(pickle_file)
print "loading finished"



#### PREPARE TRAINING DATA
train = []
def add_file_to_train(file, cat):
	article = re.sub('[—“”’&…\n\"%£\'–‘\"]', ' ', file.read()).encode('utf-8')
	sentences = article.split('. ')
	for s in sentences:
		train.append((s, cat))

def add_line_to_train(line, cat):
	print line
	article = re.sub('[—“”’&…\n\"%£\'–‘\"]', ' ', line).replace('é', 'e').replace('\s', ' ').encode('utf-8', 'ignore')
	sentences = article.split('. ')
	for s in sentences:
		train.append((s, cat))

real_path = _path + "/real"
reals = [f for f in listdir(real_path) if isfile(join(real_path, f))]
for real in reals:
	file = open(real_path + "/" + real, 'r')
	add_file_to_train(file, "real")


fake_path = _path + "/beaverton"
fakes = [f for f in listdir(fake_path) if isfile(join(fake_path, f))]
for fake in fakes:
	file = open(fake_path + "/" + fake, 'r')
	lines = file.read().split('\n')
	for line in lines:
		if line:
			add_line_to_train(line, "fake")


###### INITIAL TRAINING OF CLASSIFIER
print "training classifier at " + str(datetime.datetime.utcnow())
cl = NaiveBayesClassifier(train)

print "finished training at " + str(datetime.datetime.utcnow())


##### UPDATE CL
# cl.update(train)


########
test = []

onions = [
    "Brainstorming the wondrous features and amenities as they came to him in a flash of inspiration, President Donald Trump on Wednesday accidentally recorded over the tape containing his meetings with fired FBI Director James Comey with an idea for a candy hotel.  There could be a revolving door made out of peppermint swirl, and then you walk in, and there s a giant lobby with chocolate fountains, peanut brittle columns, and beautiful rock candy chandeliers,  said Trump into a handheld cassette recorder, replacing over 90 minutes of crucially important conversations that could be used as key evidence in determining whether obstruction of justice occurred with musings about a gumdrop garden and olympic-sized pudding pool.  The ballroom could have a hard caramel floor with ornate wall designs carved into stucco made from nougat, and then red taffy curtains and a Jolly Rancher piano oh, and all the beds would be made out of marshmallow, and the sheets could be cotton candy.  After realizing his mistake, Trump reportedly became paranoid that if the FBI got access to the tapes, they could steal his idea.",
    "Quickly crumpling up all 500 pages of the legislation upon hearing footsteps in the hallway, sources reported Tuesday that a panicked Senate Majority Leader Mitch McConnell shoved the entire Senate healthcare bill in his mouth as a Democratic senator walked past. According to witnesses, McConnell became visibly flustered upon realizing there was no place to hide from the Democratic colleague approaching his doorway and began ripping wads of documents from a binder and cramming them through his open jaws as rapidly as possible. Asked about the location of an upcoming meeting, McConnell, cheeks distended to many times their original size, reportedly grunted several times and gestured toward a nearby conference room. At press time, McConnell had spit out the massive clump of saliva-coated, half-chewed papers, which, while largely illegible, would reportedly insure 10 million more people than the original.",
    "Citing the poor quality of both the design and craftsmanship, members of the Hunter family told reporters Friday that the home s versatile game table could be easily converted to play small, shitty versions of pool, air hockey, and foosball.  Right now it s a tiny, cramped foosball table, but if you want to play air hockey on a chintzy rink that doesn t blow any air at all, then all you have to do is just flip it right over,  said Jeffrey Hunter, 14, noting that the miserable excuse for an air hockey table came equipped with two strikers too small to grip properly and a single puck the size of a casino chip.  It s got a cheap pool table component that comes with 18-inch billiard cues and shitty miniature balls that are impossible to hit accurately, so you can also play a game that barely resembles pool. This table s got whatever you might want to play for two minutes before getting completely frustrated and stopping.  At press time, the table s foosball component had reportedly become even shittier after the missing soccer ball was replaced with the eight ball.",
    "Observing that the unborn child was producing the smooth, fluid strokes expected in the third trimester, ob-gyn Dr. Theresa Umbers reportedly assured world No. 4 ranked tennis player Serena Williams at an appointment Tuesday that her fetus was developing its serve right on schedule.  As you can see on the ultrasound, your baby is getting great extension on its serve, and if you look closely you can even spot the beginning of a little topspin,  said Umbers, who noted that only a few weeks ago Williams  child had developed the ability to loosely form a western forehand grip.  Your baby s pinpoint stance is exactly where you want it to be at 24 weeks. Pretty soon it will be able to generate more power for aggressive serves, and you might even be able to feel its follow-through in the next few days.  According to sources, Williams has chosen to wait until the birth to learn whether her child is a baseline or serve-and-volley player.",
    "Saying the condiment was really putting the rest of the team on its back, area man Kevin Bentley confirmed Thursday that the chipotle mayo was doing all the heavy lifting in his sandwich.  Looks like this mayo is going to have to carry us across the finish line, because there s absolutely nothing else on this sandwich that has anything going for it,  said Bentley, explaining that the spicy southwestern spread would, as usual, have to lead the charge since the roast turkey had zero to contribute, and the shredded lettuce was essentially dead weight.  The soggy tomato sure as hell isn t helping, and that single slice of swiss cheese might as well have not shown up today. The bread should be pitching in a lot more, but it s just sitting there like it knows the chipotle mayo is going to bail it out eventually which it will, of course, just like always.  At press time, realizing that even chipotle mayo couldn t prop up every sandwich on its own indefinitely, Bentley tried easing its burden by adding some crushed potato chips. ",
    "Noting that some discomfort should be expected while traveling to a faraway place in just a few goddamn hours, officials from ultra-low-cost carrier Frontier Airlines reportedly told customers Thursday to just fucking deal with it.  I get that you re cramped and miserable, but if you just shut the hell up and sit there for a few goddamn hours, you ll soon be at your destination,  said CEO Barry L. Biffle, urging passengers to suck it up and quit whining so the flight could get on its merry fucking way.  Who gives a shit if you have no leg room and the seats are stiff? Soon you ll be 800 miles from where you are now, and it ll be like the last two hours of your life never even fucking happened. You re the ones who wanted to save $150, so you re welcome, assholes.  Biffle added that he didn t want to hear any bitching and moaning about wanting in-flight food options, because everyone can just stuff their stupid faces when they land."
]

truths = [
	"French President Emmanuel Macron has taken another swipe at Donald Trump over the US president's policy on climate change - this time backed up by the muscle of Arnold Schwarzenegger. In a video on social media, Mr Macron is joined by the Terminator star as he vows to  make the planet great again . Make America great again  was Mr Trump's presidential campaign slogan. Mr Macron has been critical of the US president's decision to withdraw from the 2015 Paris climate agreement. Could this latest development, exposed in a post on Twitter on Friday, be the start of a new  political bromance ? Speaking into his phone camera, Mr Schwarzenegger said that he and Mr Macron had been  talking about environmental issues and a green future  together. The footage was posted on the social media site with the former film star and California governor saying he was  truly honoured  to meet Mr Macron, adding that the pair would  work together for a clean energy future . The 10-second clip runs for the full duration with the caption:  With President Macron, a great leader! In April, Mr Schwarzenegger received France's Legion of Honour in recognition of his environmental work. He said that he felt  very honoured  and that  we have created the mess and now we have to get rid of the mess... it doesn't matter who is president . When Mr Trump announced earlier this month that the US was withdrawing from the 2015 Paris climate agreement, Mr Macron said in a statement that the decision was  a mistake for the US and for our planet . I tell you firmly tonight: We will not renegotiate a less ambitious accord. There is no way. Don't be mistaken on climate; there is no plan B because there is no planet B,  he said. The Paris climate agreement was established to limit the impact of carbon emissions on the environment, with countries committed to keeping the rise in global temperatures  well below  2C.",
	"Mr Trump said Mr Mueller's friendship with James Comey, who had been heading the inquiry until sacked from his role as FBI chief, was  bothersome . Asked on Fox News whether Mr Mueller should step down, Mr Trump said:  We're going to have to see. However, Mr Trump did call Mr Mueller an  honourable man . Mr Mueller was given the role of special counsel by the justice department to lead its investigation into alleged Russian interference after Mr Comey was sacked on 9 May. Mr Mueller has not given any details of his investigation but US media have reported he is investigating Mr Trump for possible obstruction of justice, both in the firing of Mr Comey and whether Mr Trump tried to end an inquiry into sacked national security adviser Michael Flynn. President Trump has repeatedly denied any collusion with Russia, calling it a  witch hunt . He did so again in his interview with Fox & Friends on Friday, saying  there has been no obstruction. There has been no collusion. He called the accusations of obstruction of justice  ridiculous . Asked whether Mr Mueller should recuse himself from the inquiry because of his friendship with Mr Comey, Mr Trump said:  Well he's very, very good friends with Comey which is very bothersome. But he's also... we're going to have to see. He also said that  the people that've been hired were all Hillary Clinton supporters . When Mr Mueller was appointed Mr Trump was said to be furious, but the special counsel won widespread initial praise from both Republicans and Democrats. However, lately some influential conservatives have intensified their attacks, openly calling for Mr Mueller's dismissal. Trump advocate Newt Gingrich urged the president to  rethink  Mr Mueller's position, saying:  Republicans are delusional if they think the special counsel is going to be fair. The New York Times has reported that Mr Trump has considered firing Mr Mueller but has so far been talked out of it by aides. Ten days ago, White House spokeswoman Sarah Huckabee Sanders said:  While the president has every right to  fire Mr Mueller  he has no intention to do so . On Friday, her colleague Sean Spicer repeated there was  no intention  to dismiss Mr Mueller. And in his Fox interview, Mr Trump said:  Robert Mueller is an honourable man and hopefully he'll come up with an honourable conclusion. Earlier this month, Mr Comey testified to Congress that Mr Trump had pressured him to drop the investigation into Mr Flynn. Mr Flynn was sacked in February for failing to reveal the extent of his contacts with Sergei Kislyak, the Russian ambassador to Washington. Mr Comey testified he was  sure  Mr Mueller was looking at whether Mr Trump had obstructed justice. US media said Mr Mueller was also examining whether Mr Comey's sacking was an attempt by the president to alter the course of the investigation. On 16 June, Mr Trump sent out a tweet appearing to accept he was under investigation, although later his aides suggested that was not the intention. On Thursday, Mr Trump also made it clear that he had not made secret recordings of his conversations with Mr Comey, despite an earlier hint to the contrary. His tweet came a day before he was required by Congress to hand over any such tapes. Mr Trump had kick-started speculation of the recordings in a tweet he posted days after firing Mr Comey, saying:  James Comey better hope there are no 'tapes' of our conversations. Allegations of collusion between the Trump team and Russian officials during the election have dogged the president's first five months in office. US investigators are looking into whether Russian cyber hackers targeted US electoral systems in order to help Mr Trump win - something Moscow has strongly denied. Separately on Friday, a Washington Post article said the Obama administration had been made aware by sources within the Moscow government last August of President Vladi­mir Putin's direct involvement in the cyber campaign to disrupt the election. The article said the administration debated a response for months before expelling 35 diplomats and closing two Russian compounds. Mr Obama had also approved planting cyber weapons in the Russian infrastructure, the article said, but the measure was not put into action."
]

#####TESTS

def add_to_test(articles, cat):
	for article in articles:
		sentences = article.split('. ')
		for s in sentences:
			test.append((s, cat))

add_to_test(onions, "fake")
add_to_test(truths, "real")

# print "testing classifier at " + str(datetime.datetime.utcnow())
# print "Accuracy: " + str(cl.accuracy(test))
# print "finished testing at " + str(datetime.datetime.utcnow())

########
def is_fake_or_real(article):
	blob = TextBlob(article, classifier=cl)
	print blob.classify()

print "onions:"
for onion in onions:
	is_fake_or_real(onion)

print "truths:"
for truth in truths:
	is_fake_or_real(truth)
########
print "pickling"
with open(pickle_file_name, 'wb') as pickle_file:
	pickle.dump(cl, pickle_file)

print "pickling finished"