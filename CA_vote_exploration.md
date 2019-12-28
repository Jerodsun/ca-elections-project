# California 2016 and 2018 Vote Exploration

_© Jerod Sun 2020. This is a large project with work in progress._

### Objective: Get insight into voting characteristics and patterns from precinct-by-precinct election results in 2016 and 2018 combined with census data. Propose questions and targeted campaign strategies based on analysis. Build classification and/or regression models on statistically significant relationships.


#### In 2020, the US will conduct a decennial census. This will provide an up-to-date comparison to the old 2010 data. ESRI (ArcGIS) has provided estimates. Block groups will change; however, the business logic should be idempotent.

#### [Vote-by-mail will soon become standard](https://www.latimes.com/opinion/op-ed/la-oe-kousser-mcghee-romero-elections-vote20190531-story.html) in California

## Overview:

To what extend do individual candidates and campaigns persuade voters? On the precinct-level, what kind of voters are more or less likely to vote a split ticket? What about turnout? The results of the last two statewide elections in California provide valuable insight into these questions.

The 2016 and 2018 statewide elections in California featured several unique characteristics. In both years, two Democratic candidates advanced from the blanket primary to the general election for a statewide election. 

- In the 2016 Senate race, California Attorney General Kamala Harris (D) won against Rep. Loretta Sanchez (D). 

- In the 2018 Lieutenant Gubernatorial election, former Ambassador Eleni Kounalakis (D) won against State Sen. Ed Hernandez (D). 

- Also in 2018, Steve Poizner, previously elected Insurance Commissioner as a Republican in 2006, ran as an independent. While Gavin Newsom easily defeated John Cox in the gubernatorial election, Poizner ran a much closer race. 

### Voting in California: 

Apart from Presidential races, California has a [top-two primary system](https://ballotpedia.org/Top-two_primary). Therefore, it is possible for two candidates of the same party to compete in the general election.

California also has an early voting period and absentee voting. California has minimal voter ID laws. All citizens can choose to vote by mail by simply checking a box when they register to vote.

In 2018, several Republican House candidates, such as incumbents Mimi Walters and David Valadao, lead on election night, but absentee ballots postmarked on or before election day ultimately broke the race against their favor. 

## Preliminaries:

Voters are individual decision-makers. Much like financial markets, past performance does not gurantee future results. Quantitative analysis into political voting patterns are useful for analyzing the _result_, not the _decision-making process_. Patterns and macro trends are quantitative - the conclusions and actions to be taken from them are not so obvious.

### Questions:

- Is there a voting pattern of Hispanics favoring the Hispanic candidate in single-party races? What about other demographic groups?

- What does the visualization of population density vs. candidate percentage look like?

- Counties have evolved quite differently from their original demographics. On the county level, are there any significant trends that differ between counties? Any where linear regression can be applied on each to calculate slope?

- What kind of precincts had the most split-ticket voters? Are there any trends with the delta, normalized for down-ballot drop-off, with any of the census data? Statistically, are these the district types with the most swing voters and candidates and parties should spend the greatest effort in targeting?

- There are 2075 precincts in San Diego County, and 628 census tracts. Is there enough data to draw certain conclusions on that subset?

In order to answer these questions, I shall use:
- QGIS / ArcGIS
- Excel

In Python:
- Pandas
- Matplotlib
- Scikit-Learn
- Keras/Tensorflow (if needed)

### Data Sources:

#### Census Shapefiles:
- Direct from the Census Bureau: `tl_2017_06_tract` [link](https://www.census.gov/cgi-bin/geo/shapefiles/index.php)
- Arcgis __Block Group__ Census Data (comes with a subset of demographic data) converted into Shapefiles.

#### Census Data:
- 2017 ACS Census Data from [IPUMS](https://data2.nhgis.org/) Block Group & Tract (archive version A)
- Alternative: ArcGIS __Block Group__ Census Data

There are 8,507 Census Tracts and 23,194 Block Groups in California. Block groups are the smallest tranche for which data is published.

#### Precinct Shapefiles:
- [DataDesk (LATimes) 2016 Github Repo](https://github.com/datadesk/california-2016-election-precinct-maps)
- [California Statewide Database](https://statewidedatabase.org/d10/g18_geo_conv.html)

#### Precinct Vote Counts:
- [DataDesk (LATimes) 2016 Github Repo](https://github.com/datadesk/california-2016-election-precinct-maps)
- [MIT Election Data and Science Lab Repo 2018 Election Results](https://github.com/MEDSL/2018-elections-official)

#### Voter Registration Data:
- [California Statewide Database](https://statewidedatabase.org/d10/g18.html)


### Business Logic:

QGIS Business Logic is provided in the 

Different projection systems.

From the raw data, geolocation matching 

There are three primary ways I gathered data:


I had already conducted area-level analysis in QGIS before finding the California Statewide Database's [Precinct Data Conversion](https://statewidedatabase.org/d10/g18_geo_conv.html) resource; however, it appears that they largely used the same system. And it seems like my census conversion is more accurate. Either way, precincts with less than 2% coverage are guranteed 

The precinct conversion logic 


Tract-level data resulted in an undercount.

Parse results:
clinton   8358011.25
trump     4355114.25
harris    7201545.97
sanchez   4553547.09

Actual:
clinton    735746
trump      477766
harris     7542753
sanchez    4701417

Total slippage:
(8753788 - 8358011) / 8753788 = 0.04521208418
(4483810 - 4355114) / 4483810 = 0.02870237588


Block-level data resulted in an overcount.

Total slippage:
(8916878 - 8753788) / 8753788 = 0.0186307916070163
(4576116 - 4483810) / 4483810 = 0.020586510133123393


# Election Results in 2016 and 2018

## Preliminaries: 2018 Election Results

[This](https://ballotpedia.org/California_official_sample_ballots,_2018) is what a sample ballot looks like.

In general, down ballot races have less votes than top-of-ticket 

#### Results section - cleanup

General election

Party | Candidate | Votes | Percentage
--- | --- | --- | ---
Democratic | Gavin Newsom	| 7,721,410	| 61.9
Republican | John H. Cox |	4,742,825 | 38.1

Total votes	12,464,235	100.0

<!-- Insert the vote counts here in a prettier format - use excel? -->

First question is voter drop off among two choices - party line voters and single issue voters.


## Assumptions: 2018 Election Results

For the purposes of this exploration, we assume that there are three categories of voters:
- Those who vote in every category
- Those who do not vote for one or more down-ballot race
- Those who do not vote for a gubenatorial candidate, but vote on one or more down-ballot race

This analysis will focus on the first and second category of voters.

The voting patterns of Republican-leaning voters will be particularly interesting, given the significant voting abstensions in offices that only had Democratic candidates on the general ballot.

Due to the mail-in ballot,  
The gubenatorial race is generally the

The 2016 Senate election also featured two Democratic candidates on the general ballot.
The 2012 Senate election is the most recent one in which a Democratic and Republican candidate qualified for the general election.

For comparison: The 2010 midterm election featured a gubernatorial and Senate race with both parties fielding a candidate.

Senate:
Boxer: 5,218,137
Fiorina: 4,217,386
Combined: 9435523

Total turnout: 10,000,160

Brown: 5,428,458
Whitman: 4,127,371
Combined: 9555829

Total turnout: 10,095,485

Jerry Brown was a former governor of California, running again under non-consecutive term limits.
