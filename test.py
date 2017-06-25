# -*- coding: utf-8 -*-
from textblob import TextBlob
import os
import re
import datetime
from os import listdir
from os.path import isfile, join
import cPickle as pickle

from textblob.classifiers import NaiveBayesClassifier

print "loading pickle"
pickle_file_name = "final_model.pkl"
# with open(pickle_file_name, 'rb') as pickle_file:
# 	cl = pickle.load(pickle_file)
print "loading finished"

print "testing"
test_real1 = '''WASHINGTON — CIA Director Mike Pompeo says he thinks disclosure of America’s secret intelligence is on the rise, fueled partly by the “worship” of leakers like Edward Snowden.

“In some ways, I do think it’s accelerated,” Pompeo told MSNBC in an interview that aired Saturday. “I think there is a phenomenon, the worship of Edward Snowden, and those who steal American secrets for the purpose of self-aggrandizement or money or for whatever their motivation may be, does seem to be on the increase.”

Pompeo said the United States needs to redouble its efforts to stem leaks of classified information.

“It’s tough. You now have not only nation states trying to steal our stuff, but non-state, hostile intelligence services, well-funded — folks like WikiLeaks, out there trying to steal American secrets for the sole purpose of undermining the United States and democracy,” Pompeo said.

Besides Snowden, who leaked documents revealing extensive U.S. government surveillance, WikiLeaks recently released nearly 8,000 documents that it says reveal secrets about the CIA’s cyberespionage tools for breaking into computers. WikiLeaks previously published 250,000 State Department cables and embarrassed the U.S. military with hundreds of thousands of logs from Iraq and Afghanistan.


There are several other recent cases, including Chelsea Manning, the Army private formerly known as Bradley Manning. She was convicted in a 2013 court-martial of leaking more than 700,000 secret military and State Department documents to WikiLeaks while working as an intelligence analyst in Iraq. Manning said she leaked the documents to raise awareness about the war’s impact on innocent civilians.

Last year, former NSA contractor Harold Thomas Martin III, 51, of Glen Burnie, Maryland, was accused of removing highly classified information, storing it in an unlocked shed and in his car and home. Court documents say investigators seized, conservatively, 50 terabytes of information, or enough to fill roughly 200 laptop computers.

Pompeo said the Trump administration is focused on stopping leaks of any kind from any agency and pursuing perpetrators. “I think we’ll have some successes both on the deterrence side — that is stopping them from happening — as well as on punishing those who we catch who have done it,” Pompeo said.

On other issues, Pompeo said:

— North Korea poses a “very real danger” to U.S. national security. “I hardly ever escape a day at the White House without the president asking me about North Korea and how it is that the United States is responding to that threat. It’s very much at the top of his mind.” He said the North Koreans are “ever-closer to having the capacity to hold America at risk with a nuclear weapon.”

–Pompeo said U.S. national security also is threatened by Iran, which he described as the world’s largest state sponsor of terror.

“Today, we find it with enormous influence, influence that far outstrips where it was six or seven years ago,” said Pompeo, a former Republican congressman from Kansas. “Whether it’s the influence they have over the government in Baghdad, whether it’s the increasing strength of Hezbollah and Lebanon, their work alongside the Houthis in Iran, the Iraqi Shias that are fighting along now the border in Syria — certainly the Shia forces that are engaged in Syria. Iran is everywhere throughout the Middle East.”'''

test_fake1 = '''An appellate court in Arizona ruled that DREAMers cannot receive in-state tuition. Each state can determine whether it can give these tax-supported benefits, the court held.
Under the ruling, illegal immigrants that have Deferred Action for Childhood Arrivals (DACA) status may not pay the in-state tuition rate.

“The decision is key because in-state tuition is yet another benefit that acts as a magnet for illegal aliens choosing to make the reckless and irresponsible decision to bring their children illegally across our sovereign borders,” Immigration Reform Law Institute’s (IRLI) executive director and general counsel, Dale Wilcox told Breitbart Texas.

Arizona voters passed Proposition 300 (“Prop 300”) in November 2006 which incorporated federal law to prohibit these postsecondary education benefits to non-qualified aliens who are residents of the state.

While the federal Personal Responsibility and Work Opportunity Reconciliation Act (PRWORA) passed by Congress in 1996 generally allows the states to determine what public benefits are available for illegal immigrants, the Illegal Immigration Reform and Immigrant Responsibility Act (IIRIRA) “does not allow any state to provide non-qualified aliens with postsecondary education benefits based upon their residence within the state.”

The appellate court noted that the Obama Administration decided to defer deportation of illegal aliens who entered the country as children – the Deferred Action Against Childhood Arrivals (DACA) program. One of the defendants in the lawsuit, the Maricopa County Community College District Board, began accepting employment authorization documents (EADs) issued by the U.S. Department of Homeland Security (DHS) as evidence that they qualified for residence-based, in-state tuition.

“The decision is certainly a victory for those Americans who appreciate and understand what a law-and-order society really looks like,” the general counsel for IRLI told Breitbart Texas.

Wilcox added, “Disturbingly, there have been several courts that have attempted to codify DACA-recipients as somehow being a lawful and benefits-eligibility class of persons, much like citizens and legal residents. But this is absolutely not the case when one looks honestly at our democratically-enacted laws.”

“DACA-recipients are illegal aliens, and they are still absolutely removable under the law,” the immigration expert and lawyer said.

The vice president of the Arizona DREAM Act Coalition, Korina Iribe, is one of the approximately 28,000 DACA recipients in Arizona, KJZZ reported.

She told Maricopa Community Colleges’ KJZZ, “It’s a road block.”

“I know that we’re going to continue to band together and fight to make sure that we continue to have in-state tuition,” she stated.

A spokesman for the college district was reported to release a statement saying, “The Maricopa County Community College District is built on a foundation of providing access to higher education for diverse students and communities, and we continue to be committed to that mission.”

Arizona Court of Appeals, Division One’s Presiding Judge Kenton D. Jones wrote the majority opinion. It reversed the opinion of the lower Superior Court in Maricopa County. Republican Attorney General Mark Brnovich appealed the trial court’s orders denying the State’s motion for judgment on the pleadings and granting summary judgment in favor of the college district.

The Supreme Court of Arizona is the next higher appellate court and the college district and other plaintiffs can appeal to that court.

After the ruling, @OneArizona, a non-partisan coalition of 19 organizations “dedicated to Latino voter registration, immigration, economic justice and education,” tweeted, “We have been brave before and we will be brave again and again.”'''

test_fake2 = '''TORONTO – During tonight’s closing ceremonies for the Pan American games in Toronto, the event’s headlining act, Kanye West, took to the stage before his cue to berate the Pan Am Games for not giving a gold medal to Beyonce.

“I respect what you do being the whitest dude on Earth, but I’m-a let you finish,” said West as he cut in front of Pan Am Chair David Peterson at the podium.

“I’m just saying that Beyonce had the greatest javelin toss of all time.”

Minutes later, Kanye West tweeted a simple hashtag, #GoldForQueenB, which has subsequently been retweeted over 700 times.'''

tests = [
	(test_real1, "real"),
	(test_fake1, "fake"),
	(test_fake2, "fake")
]

for test in tests:
	tb = TextBlob(test[0].decode('utf-8'))
	test_x = np.array([tb.sentiment.polarity,tb.sentiment[1]])
	test_x = test_x.reshape(1, -1) 
    test_pred = model.predict(test_x)
    print test[1]
    print test_pred
# tb = TextBlob(test_fake2.decode('utf-8'))
# test_x = np.array([tb.sentiment.polarity,tb.sentiment[1]])
# test_x = test_x.reshape(1, -1) 
# test_pred = model.predict(test_x)

# print test_pred
	
print "testing finished"

print "pickling"
with open(pickle_file_name, 'wb') as pickle_file:
	pickle.dump(cl, pickle_file)

print "pickling finished"