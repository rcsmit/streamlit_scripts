
# # -*- coding: utf-8 -*-
# """
# Created on Thu Oct 22 12:04:30 2020

# @author: rcxsm
# """

# # -*- coding: utf-8 -*-
# """
# Created on Wed Oct 21 23:40:39 2020

# @author: rcxsm
# """

# # Leela game
# # https://www.maa.org/sites/default/files/pdf/upload_library/22/Polya/Cheteyan-2012.pdf

# """

# https://irgp2.ru/en/igra-lila-moi-opyt-ispolzovaniya-nastolnaya-igra-leela-lila---lila-igra/


# Rules of the game LILA:
# Participants put their symbols on cell 68 (Cosmic Consciousness).
# Sitting around the playing field, participants take turns throwing a bone,
# passing it clockwise to a neighbor.
# To enter the game, you need to throw six points, then the player goes to cell
# number 1 (Gate of Life) and from there, immediately, to cell number 6 (Misconception).
# Until the player rolls six, he remains on the cell of Cosmic Consciousness (68)
# and cannot be born in the game.

# TO IMPLEMENT
# Each time a player drops the number six (with the exception of the first six,
#                                          which led the player to his birth in the game),
#  he rolls the dice again. If six rolls again,
#  the player rolls the die again. As a result, all numbers are summed up.
#  A player takes as many steps as total points.
# Exception: if six drops out three times in a row, they are not summed,
# but are reset. The player rolls the dice again and takes as many steps
# as there were points in the fourth throw.

# TODO
# If a player rolls four or more
# sixes in a row, he continues to roll the dice until a number other than
# six falls out, and then he goes forward by the number of steps equal to the total
#  sum of all points thrown, and then passes the dice.

# DONE
# Arrows and Snakes, symbolizing Virtues and Vices on the path, accelerate the
#  player’s movement and strengthen his internal awareness.
# An arrow moves the player up. When a player hits the base of the Arrow, he
# rises up through the body of the Arrow to the cell where its tip is located.
# The snake lowers the player down. If the head of the Snake is located in the
# cage where the player got, he falls down her body down to the cage where her
# tail is located.


# DONE
# The game ends when the player enters cell 68 (Cosmic Consciousness) -
# either along the Arrow from the cell of Spiritual Devotion, or during a
# gradual ascent, for example, from cell 66, throwing a deuce.

# DONE
# If the player reaches the eighth row, but, without falling on
#  the cell of Cosmic Consciousness (68), goes further and stops
#  at the cells with numbers 69, 70 or 71, he must wait until
#  either the exact number of steps that separates him from the cell
#  72 falls out ( Darkness), or a small number that allows him to advance
#  two or one step forward (for 69, 1, 2 or 3 are suitable, for 70 - 1 or 2,
#                           for 71 - only one).
#  In this case, the player needs to go down to cell 51 (Earth) using the
#  Snake on cell 72 (Darkness).


# If a player enters the Absolute cage (69), he must wait until
# the serpent of Darkness (72) returns him to the ground (51)
# so that he can continue his path to Cosmic Consciousness (68)
# either by gradually moving upward, or, having thrown the three,
# through the field of Spiritual Devotion, from where the arrow
# leads the player directly to his goal. On cell 71, all numbers
#  except one are useless for him - just like on cell 67 he can no
#  longer use the six that has fallen out. If a unit falls, then he
#  enters field 68 and the game ends, but it will continue if two,
#  three, four, five, or six fall.

# -------------------------------------------------

# 100 vakken spel
# https://www.jstor.org/stable/3619261?seq=1

# #ladders
# 1,38
# 4,14
# 9,31
# 21,42
# 28,84
# 36,44
# 51,67
# 71,91
# 80,100

# #snakes
# 16,6
# 47,26
# 49,11
# 56,53
# 62,19
# 64,60
# 87,24
# 93,73
# 95,75
# 98,78
# """

import random
import sys
import time

from plotly.graph_objs import Bar, Layout
from plotly import offline
from matplotlib import pyplot as plt
from collections import Counter
import streamlit as st

squares = [
                [1, 'Birth', 'janma', 'Entrance to samsara. The six fell out - the sixth element (consciousness) is connected with five material elements (ether, air, fire, water and earth). The unit is the root of creation, an independent person, the search for its own unique path.',0, 7],
    [2, 'Illusion', 'Maya', 'The game of hide and seek begins with himself. Awareness of unity (yoga) with the Higher Consciousness is inferior to the feeling of disunity, false self-identification, ego, duality (number two). This feeling is created by maya - the illusory power of the Higher Consciousness. Maya is the tail of the serpent of darkness (tamas).',1, 7],
    [3, 'Anger', 'krodha', 'Anger is the first and most base manifestation of the ego when obstacles arise in the realization of one`s false “I” and, as a result, insecurity. Anger is the tail of the snake of Egoism. Anger lowers to the level of the first chakra and burns all achievements, depriving the mind. Anger can also purify and develop perseverance and firmness of character, if its cause is not selfishness and it is not directed at anyone specifically. The three symbolizes dynamism, and the dynamics can be both negative and positive.',2, 7],
    [4, 'Greed', 'lobha', 'Selfishness, dissatisfaction and insecurity give rise to greed, money-grubbing, the desire to secure their place in the Game. Greed is the tail of the snake of jealousy. Material support is the main concern of the player at the level of the first chakra. Even with everything you need, such a player still feels empty. And to drown out this feeling, he is trying to take possession of more and more material wealth. Greed makes a person shortsighted, but can also be aimed at love, the acquisition of spiritual experience and knowledge, which can work for the good. Four symbolizes completeness. The desire for completeness at this gross material level turns into greed.',3, 7],
    [5, 'The physical plane', 'bhu-loka', 'Earthly, the roughest plane of being, the first of seven locks (levels of existence). On this plane, the player is completely absorbed in material achievements related to the body. If the problems associated with the physical plan are not resolved, you will not be able to rise higher. There are no ascending arrows from this dense plane, and the tails of seven snakes lead here, lowering the player from other, higher planes. Pass and gradually implement this plan of being will have all the players. Five is the number of material elements: ether, air, fire, water, earth. There are also five facets of the senses, organs of gross interaction with matter: ears — for sound, skin — for touch, eyes — for perception of shape and color, tongue — for sensation of taste and nose — for smell.',4, 7],
    [6, 'Delusion', 'moss', 'The fallacy of kinship, dependence, obsession, which obscures the gaze of the player, preventing him from seeing things as they are, and causing them to reborn. Unlike maya, which is the world of illusion itself, delusion is what binds the player to this world. Misconception stems from a misunderstanding of the laws of the Dharma, World Order, Ecumenical Religion, the Laws of the Game. Misconception is the first cell that a player enters after he rolls the six. The tail of the serpent of godlessness or lack of true religiosity also leads into this field. The six is \u200b\u200bassociated with creative activity and is in balance, but this illusory balance can become an obstacle to moving on, creating the illusion of comfort within the framework of materialism, egoism and sensual pleasures. Those in this field continue to suffer from anger and greed.',5, 7],
    [7, 'Vanity', 'Mada', 'Vanity is self-deception, pride, intoxication with imaginary or genuine greatness and virtues. Vanity is the result of bad communication (the tail of a snake called "Bad company"), and bad communication is the result of bad desires. Seven is a number symbolizing the problems of adaptation, by its nature the seven is lonely and seeks to complete, to create a society around itself. This is the number of writers and artists who, in the absence of development, are in false pride and unrealistic dreams. They love the shocking and overthrow of the foundations, tend to create their own religion and spend their lives in entertainment.',6, 7],
    [8, 'Greed', 'Matsarya', 'Vanity leads to greed - a person thinks that he deserves everything that others have and even more. And he is ready to do anything to take possession of what belongs to others. Greed is strongly associated with envy - it is at the tail of the snake of envy. Greed is envy combined with greed. Eight is a number that, when multiplied by any other, decreases, and when multiplied by nine, it first increases, and then again comes to 8. The subtle decreases when the gross grows. And so on until there is penetration into the essence of the gross, and then it becomes subtle again. Knowing the gross brings wisdom and allows you to move on.',7, 7],
    [9, 'The sensual plan', 'kama-loka', 'The ninth square completes the first level and leads to the second, which begins with purification. Without going through the sensual plane, it is impossible to rise higher. Kama is a desire that is an incentive to development. But this desire is gross. The tail of the snake of ignorance or the gradual exploration of the first chakra leads here. Nine symbolizes completeness and perfection.',8, 7],
    [10, 'Cleansing', 'tapa', 'Beginning of the second level. Loss of energy on the first level creates a feeling of emptiness and dissatisfaction, even despite a sense of comfort. Then the player’s attention turns to purification. Here the arrow begins, leading to the heavenly plane and allowing you to immediately overcome all the problems of the second level.',8, 6],
    [11, 'Entertainment', 'gandharva', 'Gandharvas are celestial musicians. Their life is dedicated to entertaining others and having fun. After cleansing, they pass to this level and fill the players with inner joy and lightness, a sense of rhythm and harmony, and the ability to hear the music of the spheres. They are already free from feelings of insecurity and preoccupation with material well-being, which is characteristic of the first level.',7, 6],
    [12, 'Envy', 'Irasia', 'The first snake. It returns the player to the greed cell to the first level. This is the first fall due to insecurity and the inability to come to terms with the fact that the other can also be on the second level or even higher.',6, 6],
    [13, 'Nullity', 'antariksha', 'Antariksha is a plane located between the physical plane and the heavens (swarga-loka). Here the player is in a "suspended" state: neither in heaven, nor on earth, in fact, to nothing. The state of antariksha is the tail of the snake of negative intelligence and comes from a lack of understanding of its purpose, existential fear and a sense of instability. The player feels a sense of inner emptiness, finds no place for himself and is in constant alarm and loss. All this is a byproduct of switching to the second chakra. The player wastes his little energy on sensual pleasures and, quickly exhausted, loses interest in life and the game. This is the main problem of the second chakra. Replenishment of energy reserves allows him to move on.',5, 6],
    [14, 'The astral plane', 'bhuvar-loka', 'The astral plane is a dimension of psychic space located between the earth and heaven. This is a plan of dreams, dreams, fruitful creativity and imagination. The player has already met his material needs and now he sees that life is much more interesting and diverse than he could have imagined, being concerned about earning a livelihood. The player`s creative imagination is released, he spends time in pleasures, sexuality becomes the main means of self-expression. The player has already overcome the earthly plan, and his imagination gives him an idea of \u200b\u200bheaven. The danger is that dreams and sensuality take the player away from reality, depleting his vital energy.',4, 6],
    [15, 'Plan of fantasy', 'naga-loka', 'Naga-loka, the world of magic snakes - the abode of those who are completely immersed in their fantasies. This abode is underground. The player lives in a fantasy world, where the possibilities of life are expanded to incredible boundaries and the usual restrictions for a person do not apply. All the player’s energy is directed to the study of this world and manifests itself in the creation of works of art, new ideas and discoveries. On the astral plane, the player begins to realize his abilities, here he is completely absorbed by this dimension of psychic space.',3, 6],
    [16, 'Jealousy', 'dvesha', 'Jealousy arises from a feeling of self-doubt. Suspicion and inability to trust loved ones deprives the player of a feeling of reliability, security and returns him to the first level in the arms of greed. To regain self-confidence and tomorrow, the player needs to go through the first level again in order to deal with the causes of uncertainty.',2, 6],
    [17, 'Compassion', 'daya', 'Compassion, mercy, empathy is a divine quality that is so strong that it raises a player from the second chakra to the eighth, to the plane of the Absolute. In the strongest form, compassion is manifested when it is directed at the enemies, when the player “turns his cheek”, instead of “giving change”. Compassion and forgiveness develops not least thanks to the imagination, which allows us to imagine what the other person feels and thinks, why he is forced to act in this way and not otherwise, obeying the rules of the great Game of the universe. Compassion pushes the boundaries of the false "I", closer to the Absolute, but does not relieve karma. Therefore, the player will have to move on until he is bitten by the serpent of Tamoguna and returns him to the ground to complete his mission.',1, 6],
    [18, 'The plan of joy', 'harsha-loka', 'Here, at the end of the second level, a feeling of deep satisfaction comes to the player. A journey through envy, insignificance, jealousy ... and the world of fantasies comes to an end - the player is approaching reality. He managed to pass the first chakra, and now he does not feel fear and is completely confident in himself. Rising above sensual desires, he completed the second stage. Ahead of him is a joyful fulfillment of karma yoga and his joy has no limits.',0, 6],
    [19, 'The plan of karma', 'karma-loka', 'No matter what level the player is at, he will strive for satisfaction, moreover, at that level. In the first two chakras, this desire manifests itself as a desire for money and sex. In the third chakra, the main need is self-affirmation, the achievement of strength and influence. Fantasies give way to understanding the real situation and the law of karma, the law of interaction and retribution, which is behind the world order. Karma (action) creates fetters, but it can also destroy them.',0, 5],
    [20, 'Charity', 'given', 'It raises the player above the problems of this level, and he moves one level up to the balance plan located at the level of the heart chakra. Entering this field, the player identifies himself with the Divine, Good present in everything, and performs actions aimed at the benefit of others, without expecting any benefit for himself. These actions fill the player with joy that accompanies the rise of energies to higher levels. Charity satisfies the evolving ego and frees the player from the shackles of the third chakra.',1, 5],
    [21, 'Atonement', 'samana papa', 'Gradually, the player realizes that in the process of satisfying feelings, he could harm others, and now this bad karma interferes with his development. It is time for repentance and correction of mistakes. Repentance also helps those who have not yet adapted to the high level of the third chakra. The player atones for his mistakes, following the principles of Dharma, universal religion, and tunes in to a higher level of vibration.',2, 5],
    [22, 'The Dharma Plan', 'Dharma Loka', 'Dharma are universal principles that harmonize a world that seems chaotic. Dharma is the laws of life in the universe, following which creates the conditions for overcoming the lower planes of being. From the Dharma plan, the player rises immediately to the seventh level, in the field of positive intelligence. Dharma is originally a property of everything in this world. Ten basic qualities that distinguish those who follow the dharma: firmness, forgiveness, self-control, restraint, purity, control over the senses and organs of action, intelligence, correct knowledge, truth and lack of anger.',3, 5],
    [23, 'Heavenly Plan', 'Swarga Loka', 'The third among the seven planes of being. On this plane, thinking comes first. The creatures that inhabit this plan emit light. In the first chakra, the player seeks security and tries to take possession of many things that could support his physical existence. In the second, he explores the world of feelings and seeks pleasure. Rising to the third plan, he opens a new dimension - the image of paradise created by thought, of the world, which satisfies his understanding of happiness and satisfies his ego, the desire for self-identification and eternal life in happiness and pleasure.',4, 5],
    [24, 'Bad company', 'ku-sanga-loka', 'In search of self-identification, characteristic of the third chakra, the player is looking for a group of other people who could support him. At the same time, he risks getting into the society of people who retreat in their actions from the laws of the Dharma. This is a "bad company." The force created by the group serves as a basis for the ego and self-conceit of the player, and, stung by a snake, the player returns to the first chakra - in the field of vanity.',5, 5],
    [25, 'Good company', 'su-sanga-loka', 'A good company at the third level is a community of people helping each other realize their best qualities and expand their ego. This community gathers around a spiritual teacher with a developed fourth and fifth chakra. In contrast, a bad company usually gathers around a charismatic leader with a developed third chakra. The result of staying in a bad company is the development of self-esteem, while in a favorable society a person develops the ability to compassion. Good communication is essential for moving to higher levels.',6, 5],
    [26, 'Sorrow', 'dukkha', 'Joy is a state of expansion, and sadness is a state of contraction. In spiritual practice, sadness arises from the awareness of one`s inability to approach the divine - due to problems with the first and second chakras. But there is a way out - selfless service, the best cure for sadness.',7, 5],
    [27, 'Selfless Case', 'Paramartha', 'If charity involves actions performed from time to time, then selfless service is a constant position, a way of being. Param means "higher," and artha is the goal for which the action is performed. Everything that is done for a higher purpose is paramartha. Higher may mean God or some other idea to which the player decided to devote his life. Selfless service is the renunciation of oneself and one`s self-identification for a higher purpose. Individuality ceases to exist as a separate unit, becoming part of a larger whole. The arrow of service takes the player to the human plane, on the fifth level.',8, 5],
    [28, 'True religiosity', 'sudharma', 'Sudharma is life in harmony with the laws of the Game. Sudharma is understanding your place in the Game and following your own dharma. Sudharma is the individual path to liberation. As soon as a player begins to understand Sudharma, he becomes religious inwardly, and religion becomes a way of his life. Rituals lose their meaning, life itself turns into a constant worship. And the player becomes ready for the transition to the ascetic plan, to the sixth level.',8, 4],
    [29, 'Atheism or lack of religiosity', 'adharma', 'Adharma is the non-observance of the laws of the Dharma, an action contrary to one’s inner nature. The cause of adharmic activity is often too much self-confidence. A person who has reached a certain spiritual level can begin to consider himself a god, capable of independently determining what is good and what is bad, without taking into account the laws of the universe, common to all. Such a person is bitten by a snake of vanity and he rolls down to the first level. Avoid this allows true faith, based on an understanding of the foundations of being and humility.',7, 4],
    [30, 'Good trends', 'uttama-gati', 'Good trends are manifested in a player spontaneously if he moves in harmony with the laws of the macrocosm. While he vibrates on the three lower planes, these trends do not develop. Their growth begins only here, in the fourth chakra, since it requires the achievement of a certain degree of internal balance. Maintaining good trends is helped by breathing control, meditation (especially in the early morning), vegetarianism, sweats, scripture study, and all the virtues mentioned in this game. All this will help stabilize and reduce the waste of energy through the lower chakras.',6, 4],
    [31, 'The plan of holiness', 'yaksha-loka', 'Yakshas are ethereal creatures that live in heaven. A player who falls on the plan of holiness experiences divine grace, oneness with the Divine, and the ability to see the manifestation of His grace throughout creation. This unity transcends simple intellectual understanding and becomes a real part of everyday life. The player’s attention is drawn primarily to the comprehension of the nature of the divine existence and the presence of the Divine in all creation.',5, 4],
    [32, 'Plan of equilibrium', 'mahar-loka', 'The first three lokas serve as an arena where the jiva (individual consciousness) lives, developing in a series of new and new rebirths. In this fourth lock, the player rises above the physical level, desires and thoughts, and lives in a state of balance in the invisible world. In the heart center, male and female energy are balanced. Here, the player overcomes the intellectual understanding of the Divine that is characteristic of the third chakra, and moves towards a direct emotional experience of His presence inside his “I”, exuding love with his words and deeds.',4, 4],
    [33, 'Fragrance plan', 'gandha loka', 'Gandha-loka is the level of divine aromas. Being at this level, the player not only acquires the ability to feel the subtlest aromas of physical and metaphysical nature, but also begins to exude a pleasant aroma. His bodily secretions are no longer malodorous.',3, 4],
    [34, 'The plan of taste', 'race-loka', 'Race is pure shades of emotions, sensations, moods, tastes. When a player falls into the plan of taste, his taste in every sense of the word is refined. He becomes a connoisseur and connoisseur of good taste, recognized by all.',2, 4],
    [35, 'Purgatory', 'naraka-loka', 'Naraka-loka is the abode of the God of death, where there is cleansing from the most subtle pollution and sins. Here the player is suffering physical and moral suffering, repents, realizes the nature of unnatural sin, gets rid of negativity and affirms his dharma.',1, 4],
    [36, 'Clarity of consciousness', 'svachchha', 'Clarity of consciousness is the light that fills a player when he leaves the fourth level to enter the fifth, where a person becomes a man with a capital letter. In Sanskrit, the word svachcha means "pure, clear, transparent." This transparency is the result of the purification process through which the player passes through the Naraka Loka (human sphere). There is no place for rationality, intellectual understanding. Devotion and faith help to overcome this condition, and the player enters the level of living knowledge. All doubts go away and absolute transparency remains. The player has visited the plans of aroma and taste and is now ready to join the upward flow of energy, lifting him to the fifth chakra.',0, 4],
    [37, 'Wisdom', 'jnana or jnana', 'Pure wisdom, free from judgment and judgment, is the power that elevates a player to a plan of bliss located in the eighth row outside the chakras. A player who understands his role in the game and the nature of the actions that will enable him to fulfill this role lives in bliss. Jnana is awareness, but not final realization. Jnana is an insight into the essence, but not the essence.',0, 3],
    [38, '', 'Prana Loka', 'Prana is the life force. In Sanskrit, it is a synonym for life and the name of the breath of life that we receive with each breath. Prana is the “spirit” in the body, the connection of the soul and the physical shell. Consciousness leaves the body with prana. With the help of pranayama, yoga raises prana on the spine and reaches various siddhis - mystical perfection.',1, 3],
    [39, '', 'Apana Loka', 'Apana is, in contrast to prana, the energy that goes down and is responsible for bodily secretions. This is the main force that helps cleanse the body. The mixture of prana and apana causes the awakening of kundalini - the colossal energy contained in the base of the spine. Kundalini bestows physical immortality and supernatural abilities.',2, 3],
    [40, '', 'Vyana-loka', 'Vyana takes pranic energy in the lungs and distributes it throughout all body systems. This vital energy is responsible for the flow of blood, the secretion of glands, the movement of the body down and up, as well as for opening and closing the eyelids. Together with prana and apana, they are responsible for the normal functioning of the body. Full control over prana, apana and vyana allows the yogi to get rid of bodily influence and direct his energy to the higher chakras. The player’s understanding of the divine presence in the whole creation, acquired during a journey through the fourth chakra, makes him seek the Divine within himself, and so his attention turns to the processes occurring in his body and reflecting the processes of the macrocosm. His consciousness and experience expand, erasing the line between internal and external.',3, 3],
    [41, 'The human plan', 'jana-loka', 'Jana-loka (or jnana-loka, the realm of wisdom), the fifth plan of the fifth level, is the abode of the Siddhas (creatures that have reached a high degree of development and have powers that allow them to perform actions that are “supernatural” for people on the lower planes) and saints constantly immersed in the contemplation of Lord Visnu. The element of air prevails here, and the bodies of beings on this plane are composed of pure wisdom that is not subject to desires. On this plane, the player comprehends the true meaning of what it means to be human. Often this is achieved directly during the transition from the level of the third chakra with an arrow leading from the field of selfless service.',4, 3],
    [42, 'Agni Plan', 'Agni Loka', 'Agni or fire - fire is the source of both colors and forms that make up the essence of the world of phenomena. Fire is a gross manifestation of energy, its vehicle. The player who falls into this plane understands that his body is also only a means, an object of sacrifice. That is why in the fire they see a mediator between man and God. All religious rituals are performed in the presence of Agni as an eternal witness and accepting the sacrifice. Fire is present in all creation; nothing can be hidden from it. He is a witness who excludes any deception and self-deception.',5, 3],
    [43, 'The birth of man', 'manusya janma', 'Passing through the Agni plan prepares the player for a real birth. Conceived in the second chakra, fed and raised in the third, filled with human emotions in the fourth, the player is now ready for the resurrection of his true human being - for a second birth. He becomes a double-born, Brahman - a follower of the Absolute Truth, Brahman.',6, 3],
    [44, 'Ignorance', 'Avidya', 'Vidya means "knowledge, knowledge," and - the prefix of negation. Lack of knowledge is ignorance. Knowledge is the player’s understanding of his role in the game, wherever he is at the moment. A player who enters the field of ignorance, forgets about the illusory nature of being and becomes attached to certain emotional states or sensory perceptions. The energy of a player stung by a snake of ignorance drops to the level of the first chakra and the plan of sensuality. Loss of understanding of the nature of maya (illusion) leads to clouding of his intellect and leads to identification with certain states.',7, 3],
    [45, 'Right knowledge', 'suvidya', 'If jnana is an awareness of the truth, “right knowledge” includes awareness and behavior (practice). This combination raises the player to the eighth plan, on the field of cosmic good. And now only one step separates him from the goal. He realizes himself as a macrocosm in a microcosm, an ocean enclosed in a drop. Correct knowledge adds a new dimension to jnana, awareness of the unity of the past, present and future. The field of correct knowledge completes the passage of the fifth chakra, the fifth row of the game. At this stage, the player reaches full awareness of his unity with the cosmos, he connects with the ultimate reality and is elevated to the plan of Rudra (Shiva), the plan of the cosmic good.',8, 3],
    [46, 'Distinction', 'Viveka', 'The ability to distinguish is the ability to listen to your inner voice, which tells you what is right and what is not. Viveka could not appear in the game before. A player can get here only after passing through the field of correct knowledge (45). If a player falls on the arrow of the right knowledge, he immediately rises to the plan of the cosmic good. Otherwise, he has to resort to the ability to distinguish to determine the further course of the game. The ability to distinguish raises the plan of happiness. At the level of the sixth chakra, a person overcomes the influence of gross matter and becomes able to see the past, present and future in any direction - he opens a third eye. But the subtle influence of maya remains. The inner voice or the voice of the Lord in the heart acts as a reliable guide.',8, 2],
    [47, 'Plan of Neutrality', 'Saraswati', 'In the sixth chakra, the negative and positive influence of the chakras and energies gradually disappears and only the neutral remains. The Sarasvati plan is the kingdom of the goddess who bears the same name. Here the player is surrounded by pure music and lives in a state of vidya, knowledge. Saraswati, the deity of learning and beauty, gives him the opportunity to achieve equilibrium and be beyond the influence of the energy fields of existence. Now he can just watch the game.',7, 2],
    [48, 'The Solar Plan', 'Yamuna', 'In the sixth chakra, the player establishes a balance between the male, solar principle and the female, moon. This harmonious fusion of elements creates the “I” of the observer, which is neither male nor female, but represents a harmonious combination of both. Below the sixth chakra, the solar and lunar energies are intertwined with each other, here they merge and become one. This feeling of a single whole characterizes the plan of asceticism. Awareness of solar energy comes on the solar plane, but this energy does not affect the observer standing in the Yamuna River and sensing how the hot solar energy of creation and destruction, life and death, passes through it. Yamuna - Sister of the God of Death of the Pit',6, 2],
    [49, 'Lunar plan', 'ganga', 'The player who falls into this plane is at the source of the attractive and enveloping female magnetic energy or psychic energy, which balances the solar energy of creativity.',5, 2],
    [50, 'Plan of asceticism', 'tapa-loka', 'Just as in the fifth chakra, knowledge was of primary importance for the player, here, on the ascetic plane, all the aspirations of the player are aimed at the hard work of repentance and austerity. The word tapas means "asceticism", "mortification of the flesh", "burning." This is the practice of meditation of self-denial. Tapa-loka is inhabited by great yogis and ascetics, who have gone along a path from which there is no return, immersed in deep asceticism, whose goal is to advance up to the next level, satya-loka. The player can enter tapa-loka directly through the practice of sudharma (the plan of religiosity) in the fourth chakra or gradually, moving through the fifth chakra, developing consciousness and mastering the system of lunar and solar energy.',4, 2],
    [51, 'Earth', 'prithivi', 'Earth symbolizes the great maternal principle. This is the scene on which the mind plays its eternal game known as Leela. Here, the player understands that the earth is not just soil, but Mother Earth. The player opens up new patterns and harmonies, new ways of playing, previously hidden from him in the fog created by involvement in the lower chakras. Despite the fact that her children cut her body and burn her soul, the Earth gives them diamonds, gold, platinum. She selflessly follows the law of the Dharma and does not distinguish between high and low. That is why the earth’s field is located in the sixth chakra. We see her body, the physical plane of the first chakra. What we cannot see is her spirit, her understanding, generosity and kindness, her greatness. This is the understanding that comes to the player when he reaches the sixth chakra. If a player fails to achieve Cosmic Consciousness, he will have to return to Mother Earth again and from here start again to a higher goal.',3, 2],
    [52, 'Plan of violence', 'chimsa loka', 'The player who has reached the sixth chakra is aware of the unity of everything. Human bodies serve only as transient forms. The true essence of all players exists outside the realm of names and forms. The player knows that death is just a change of life scenario. There is a danger that the player will begin to resort to violence, realizing that his actions ultimately do no real harm to other players. But he has not yet completely freed himself from karma, and violation of the laws of dharma will entail a fall to the fourth level - in Purgatory.',2, 2],
    [53, 'The plan of liquids', 'jala-loka', 'Water is cold in nature, it absorbs heat, bringing coolness. The heat of asceticism of the sixth chakra, austerities, makes the player cruel. He needs to go through the clear waters of the liquid plane in order to extinguish the burning energy of violence and turn it into the even warmth of spiritual love.',1, 2],
    [54, 'Plan of spiritual devotion', 'bhakti-loka', 'Bhakti, or spiritual devotion, is based on the statement: "Love is God, and God is love." A bhakta devotee loves his deity. The deity is beloved, and the devotee is in love. Bhakti is the immediate method; it is the shortest path to the Divine. All yoga and all knowledge, jnana, rests on the cornerstone of true faith, true devotion and love, true bhakti. There is nothing higher than love, and bhakti is the religion of love. Jnana makes the player a sage, while bhakti turns him into a divine child on the lap of his Mother under the benevolent protection of the Father. The sage has a long way to go to meet the Lord. Bhakta is always surrounded by its Divinity, present in the myriad of its forms and names in every part of life experience.',0, 2],
    [55, 'Selfishness', 'ahamkara', 'Ahamkara is a feeling of separation of one’s existence. When all the player’s attention is directed exclusively to the satisfaction of his desires, even spiritual ones, he becomes an egoist. Satisfaction of spiritual and material desires pushes into the background love, humility, sincerity, attentiveness to others and all other dharmic qualities. Self-focus contradicts the spiritual path - the path from oneself to the Divine. As soon as obstacles arise for satisfying desires, such a player falls into the old trap of anger and falls to the first level.',0, 1],
    [56, 'Plan of primordial vibrations', 'Omkara', 'Om is the one sound present throughout the universe, manifested and unmanifest. It is the subtlest of the forms in which energy exists. Omkara is a plan of vibrations that produce this cosmic sound in harmony with all other vibrations. The player who gets here realizes that Om is a vibration that fills all the elements of being.',1, 1],
    [57, 'Gas plan', 'vayu-loka', 'Vayu-loka (literally "plan of the air") is located in the area of \u200b\u200bsatya-loka, the plane of reality, in the seventh row of the playing board. This vayu is not that wind, or air, which is on the physical or earthly plane. This is the essence of the physical element of air. Vayu-loka is a plan where a player becomes a stream of energy, with which the whole atmosphere moves, overcoming gravity. Enlightened souls with light bodies dwell here, who have not yet reached satya-loka - the plane of reality. In the sixth chakra we met with the plan of liquids, but the liquid still has a shape. Gas does not have any particular shape. Liquid has weight; gas does not. The player is no longer burdened by anything; he has gained true freedom of action. He becomes a creature that is not subject to gravity and has no form.',2, 1],
    [58, 'Plan of radiance', 'teja-loka', 'Teja is the light that was created in the beginning. The world that we perceive in an awake state is a world of phenomena and forms, a phenomenal world that exists in the light (tedzha) from which it materializes. This world is similar to the world of our dreams, but it is not. This state is completely “made” of light and truly enlightened dwell in it.',3, 1],
    [59, 'Plan of Reality', 'Satya Loka', 'Satya-loka - the last plan of the seven main locks located in the spinal column of the playing board. Here the player reaches the world of Shabd-Brahman and is on the verge of liberation from the cycle of rebirth. He has reached the highest plane, beyond which is Vaikuntha, the seat of Cosmic Consciousness. On this plane, the player is affirmed in Reality. Here the player reaches his highest chakra and himself becomes a reality, a realized being. He is in a blissful state of samadhi, like a drop in the ocean. But even here, the player has not yet reached liberation. Here, at the seventh level of the game, there are three snakes. The first is egoism. The second is negative intelligence. The third is tamas. Having reached the plan of reality, the player escaped one of these snakes, but two are waiting for him in front, challenging his desire for liberation.',4, 1],
    [60, 'Positive Intelligence', 'Subuddhi', 'Subuddhi is the correct understanding that comes only with the attainment of the plan of reality. After the player has reached satya-loka, his consciousness becomes perfect and free from duality and he comprehends the Divine in all forms and phenomena of this world. Such a consciousness is subuddhi.',5, 1],
    [61, 'Negative Intelligence', 'Durbuddhi', 'If the player does not follow the laws of the Dharma, doubting the cosmic nature of being and the divine presence in each of his experiences, he is stung by a snake of negative intelligence, dropping down to the plane of insignificance. He did not learn to accept everything that the world offers him, to use all the opportunities for development, and to see God`s hand in everything. Now he has to go through all the vibrational plans related to the second chakra, unless the arrows of compassion and charity help. If he does not resort to the help of these arrows, he will have to redeem his negativity (cell 21 - redemption) and again find the Dharma or choose a completely new course of action.',6, 1],
    [62, 'Happiness', 'Sukha', 'Happiness, or sukha, comes to the player when his consciousness tells him that he is very close to the goal, giving him confidence that he is approaching liberation. The feelings he experiences are indescribable; they cannot be described in words. He feels the happiness that a river experiences when connecting to the ocean after a journey of a thousand miles. It is a feeling of merging with its source. If, being in such a happy state, the player does not neglect his karma, does not become lazy and passive, he has a real chance to achieve Cosmic Consciousness during this life. But if he is so overwhelmed by the experience of happiness that he forgets about the need to act, feeling that his mission is close to the end, the tamas snake lurking next to him is ready to swallow him, returning his energy to the level of the first chakra.',7, 1],
    [63, 'Darkness', 'Tamas', 'In Sanskrit, the word tamas means "darkness." Tamas is the snake of darkness, the longest snake in the game, which ruthlessly pulls the player into the illusion from the radiance of the plan of reality. In the seventh chakra, tamas is the ignorance that arises from attachment to sensory perception. This ignorance comes after the player realizes a state of happiness and thinks that this is the end of the need to fulfill karma. But here the player still cannot stop all activity. He forgot that until he reaches liberation, the game is not over. Inaction is an attempt to avoid the law of karma. Tamas is an attribute of tamoguna, its manifestation in the microcosm. When the same power is discussed as an attribute of prakriti in the eighth row of the game, it is called Tamoguna.',8, 1],
    [64, 'The phenomenal plan', 'prakriti-loka', 'Manifested prakriti is a material world consisting of elements - earth, water, fire, air and ether (akasha), as well as mind (manas), intellect (buddhi) and ego (ahamkara). These are the eight gross manifestations of prakriti. Divine prakriti is Maya-Shakti, the illusory energy of God, the shadow of God.',8, 0],
    [65, 'Plan of the inner space', 'uranta-loka', 'Leaving behind the seventh row and realizing the existence of prakriti, the player begins to penetrate the source of all phenomena of the phenomenal world - the great consciousness. The player merges with him, and at that moment all duality disappears. The player receives the pure experience of immense dimensions, the infinite space lying inside his "I".',7, 0],
    [66, 'The Bliss Plan', 'Ananda Loka', 'Consciousness is described as truth, being and bliss - Satchitanananda. Ananda is the highest truth, the essence of being. In the process of creation, the "I" is gradually covered with five shells. Of these, the first and most subtle — anandamaya-kosh — is the body of pure being, pure experience of consciousness. This is the body of bliss, in the center of which is Cosmic Consciousness. During the period corresponding to creation, it acts as an individual consciousness.',6, 0],
    [67, 'Plan of the cosmic good', 'Rudra-loka', 'Rudra is one of the names of Shiva. All creation goes through three phases. The manifestation is accompanied by the maintenance of the arising form and inevitably ends with decay or destruction. These three processes - creation, maintenance (preservation) and destruction are carried out by the three forces of the Most High, who was not created by anyone, but who creates everything. Out of his will, the Creator (Brahma), the Guardian (Vishnu) and the Destroyer (Shiva) were born. These three forces are interdependent and interconnected. Creation takes place according to the will of the Lord; according to his will, creation is preserved and destroyed in the end. Without the destruction of false self-identification - the concept of a separate reality, an individual ego - true union (yoga) is impossible. Thus, Shiva, destroying the false ego, unites the individual consciousness with its cosmic source. Rudra-loka is one of the three central squares of the top row of the playing board, where there are divine forces responsible for the entire creation, which everyone seeking liberation seeks to unite with. The desire for the right knowledge leads the player to the seat of Shiva. Here he comes to the realization of the cosmic good, the essence of which is truth, and form is beauty.',5,0 ],
    [68, 'Cosmic Consciousness', 'Vaikuntha-loka', 'Towering above all the locks, beyond all limits, is Vaikuntha - the lock of Cosmic Consciousness, the life force (prana) of all manifest reality. This loka also consists of an element called Mahat, which serves as the source of all other elements. Before the start of the game, the participant accepts the importance and significance of this plan of Genesis, which will always be his goal. Vaikuntha is the abode of Vishnu, a place that every follower of Hinduism hopes to achieve by completing its existence in its current form. Here is Cosmic Consciousness, because Vishnu, being the Truth, is the patron and protector of consciousness in its ascent. Points falling on the dice of karma correspond to the level of vibration of the player. The bone determines both the player’s position on the field and the distance traveled and the path ahead. The player can follow the discipline of ashtanga yoga, the octal path, gradually passing level after level. Or, following the Dharma, become a bhakta - a spiritual devotee. All paths lead to one goal. Whatever the player’s path is among all the innumerable possibilities, he has now reached the Vishnu monastery. Vishnu serving the essence of creation, Truth. It is located directly above the plane of reality, because Truth is the highest reality. The game ends. What will happen now depends on the player. The nature of the space game is simple - it is the discovery of new combinations. With what new karmas, with what kind of fellow travelers will the player be able to re-enter the game, trying again to find a state that will be his real haven? He can continue this game of hide and seek with himself or forever remain outside the game. Or he may return back to Earth to help other seekers reach their goal.',4, 0],
    [69, 'Plan of the Absolute', 'Brahma-loka', 'Rudra is located on one side of Vaikuntha-loka, and Brahma is on the other. Together they form the triad of Brahma, Vishnu and Shiva in the center of the top row of the playing board. Those who are fortified in the truth live here, not afraid of a subsequent return to the fulfillment of karmic roles. Practitioners of mercy come here, they are in the abode of Brahma, not knowing fear. Brahma is the creator of the material world, the active principle of the noumenon, a force that transforms consciousness into countless forms and reflections. His abode is Brahma-loka. The player reaching this place merges with this absolute power, this subtle principle. Although Brahma-loka is located next to the plan of Cosmic Consciousness, Brahma cannot give the player liberation. The game must go on. Brahma determines the form of the game, but there is something else besides form. Only Truth comprehended on the path of spiritual devotion or by gradually moving up the ladder of yoga to the Highest Good (Shiva) can grant final liberation.',3, 0],
    [70, 'Sattvaguna', '', 'Sattva itself is the mode of virtue that contributes to the fulfillment of the Dharma. Sattvaguna is synonymous with concepts such as light, essence, true nature and higher levels of vibration. The calm, unperturbed state of meditation leading to samadhi is realized when sattva prevails. But the three gunas in the world of karma always exist together and sattva sooner or later becomes mixed with rajas (passion) or tamas (ignorance). Unclean sattva is always subject to the effects of karma. Pure sattva is not different from Cosmic Consciousness, Supreme Truth. Sat means "truth."',2, 0],
    [71, 'Rajoguna', '', 'Rajoguna is an activity in consciousness, or active consciousness. A player who has reached the eighth row, but who has failed in an attempt to realize Cosmic Consciousness, is carried forward by the forces of karma, activity. This activity is the cause of all suffering, it presupposes the presence of a figure who inevitably falls victim to his ambitions and the expectation of the fruits of the activity. Any obstacle to the desired goal gives rise to pain and suffering.',1, 0],
    [72, 'Tamoguna', '', 'Tamoguna hides the truth so that the rope seems to be a snake, and the snake - a rope. Darkness is the main sign of Tamoguna, and its nature is passivity. The player who gets here immediately leaves the level of cosmic forces and returns to earth to search for a new way of climbing. What happens next depends only on the player and that One, which is the Truth.',0, 0],
        ]
specials_oud= (
            #ladders
            [10,23],
            [17,69],
            [20,32],

            [22,60],
            [27,41],
            [28,50],

            [45,67],
            [37,66],
            [46,62],
            [54,68],
            #snakes
            [63,2],
            [55,3],
            [16,4],
            [29,6],
            [24,7],
            [12,8],
            [44,9],
            [52,35],
            [72,51])

specials = (#ladders
    [1,38],
    [4,14],
    [9,31],
    [21,42],
    [28,84],
    [36,44],
    [51,67],
    [71,91],
    [80,100],

    #snakes
    [16,6],
    [47,26],
    [49,11],
    [56,53],
    [62,19],
    [64,60],
    [87,24],
    [93,73],
    [95,75],
    [98,78])
results=[]

def throw_dice(numberofrounds):
    """ Throw the dice """

    number = random.randint(0,5)
    number_thrown = number + 1
    #print (f"{play}/{numberofrounds} - You have thrown { number_thrown}")
    numberofrounds +=1

    return number_thrown, numberofrounds


def give_explanation(a):
    for s in squares:
            if s[0]== a :
                if s[1]=="":
                    explanation_header.write (f"Square {s[0]}. {s[2]}")
                elif s[2]=="":
                    explanation_header.write (f"{s[0]}. {s[1]}")
                else:
                    explanation_header.write (f"{s[0]}. {s[1]} ({s[2]})")
                explanation_txt.write (s[3])

def playgame(started, number_of_rounds, current_position):
    from random import random

    random_number = str(int(100000* random()))
    st.write(random_number)
    if st.button ("Roll dice x", key = random_number):
        st.write("I roll the dice")
        st.stop()

        after_rolling_dice(started, number_of_rounds, current_position)
    else:
        st.write("Dontknowhwy")

def after_rolling_dice(started, numberofrounds, currentposition):

    thrown, numberofrounds = throw_dice(numberofrounds)

    st.write (f" {numberofrounds}. You have thrown {thrown}")

    # TO DO
    # Exception: if six drops out three times in a row, they are not summed,
    # but are reset.  If a player rolls four or more
    # sixes in a row, he continues to roll the dice until a number other than
    # six falls out, and then he goes forward by the number of steps equal
    # to the total sum of all points thrown,
    # and then passes the dice.

    if (started == False and thrown == 6) or (started == True):

        if (started == False and thrown == 6):
            started = True

        oldposition = currentposition
        currentposition += thrown

        if currentposition > 72:
            currentposition = currentposition - thrown-tempthrown
            st.write (f"You stay at {currentposition}")
        else:
            st.write (f"             ({oldposition} +  {thrown} = {currentposition})\n ")
            give_explanation (currentposition)
        for value in specials:
                if currentposition == (value[0]):
                    currentposition = value[1]
                    if value[0]>value[1]:
                        st.write(f"Oh no!!!! You go from {value[0]} to {value[1]}\n")
                        give_explanation(value[1])
                        break
                    else:
                        st.write(f"Whoop whoop!! You go from {value[0]} to {value[1]}\n")
                        give_explanation(value[1])
                        break

        if currentposition == 68:   #COSMIC CONSCIOUSNESS
            st.write ("YOU MADE IT")


        else:
            playgame(started, numberofrounds, currentposition)
    else:
        # Player didn't throw a 6 yet to start
        st.write ("You have to throw a 6 to start")
        playgame(started, numberofrounds, currentposition)


def main():

    st.title ("Leela game")
    global you_have_thrown, explanation_header, explanation_text, you_move, press_enter, button_place
    button_place =st.empty()
    you_have_thrown = st.empty()
    explanation_header = st.empty()
    explanation_text = st.empty()
    you_move = st.empty()
    press_enter = st.empty()
    started = False
    playgame(started, 0,0)

global numberofrounds
global currentposition
global play
global numberofgames

global printregel
global started
global tempthrown
global zesnumber_thrown

global r
global printregel


global started

numberofgames = 1

numberofrounds = 0
currentposition = 0
tempthrown = 0
zesnumber_thrown = False
started = False

r = 1
printregel=[]
debug = True
#explanation = True


# https://irgp2.ru/en/igra-lila-moi-opyt-ispolzovaniya-kosmicheskaya-igra-dzhagat-lila/

main()


