#!/usr/bin/env python

import os
import re
import argparse
import traceback
import sys

import urllib.request

from bs4 import BeautifulSoup
import pandas
from urllib.request import Request, urlopen

def url_generator():

  alphabet = 'AA'
  suffix = 'ABCD_'
  counter = 1
  while True:
    value = '{}{}{:03d}'.format(suffix,alphabet,counter)
    if counter == 999:
      counter = 1

      if alphabet == 'ZZ':
        raise Exception('End of sequence')

      if alphabet[1] == 'Z':
        alphabet = '{0}A'.format(
          chr(ord(alphabet[0]) + 1)
        )
      else:
        alphabet = '{0}{1}'.format(
          alphabet[0],
          chr(ord(alphabet[1]) + 1),
        )
    else:
      # Increment counter
      counter += 1

    yield value

def scrape_url(url):

  try:

    # Fetch page
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request) as response:
      html = response.read()

    # No result
    if not html:
      raise Exception('Unable to fetch data')

    # Parse page
    soup = BeautifulSoup(html,features="html.parser")

    # Search for main container
    main = soup.find("div", {"class": "main-elmt"})
    if not main:
      raise Exception('Unable to find main element')

    # Extract data from table
    result = {}
    for row in main.findAll('tr'):
      header = row.find('th')
      data = row.find('td')

      if not header or not data:
        continue
      else:
        href = []
        for a in row.findAll('a'):
          href.append(a['href'])

        # Parse entry
        header = header.text.lower().replace(' ', '_')
        data = data.text
        result[header] = data
        if len(href)!=0:
          result[data] = href[0]
        else: 
          result[header] = '-'

    return result


  except Exception as e:
    return None
  except urllib.error.URLError as e:
    print('URL {} does not exists'.format(url))
    return None

def find_url(links, url):

  list_url = []
  for link in links:
    if link is not None:
      if link.startswith(url):
        list_url.append(link)
  
  return list_url
     

if __name__ == "__main__":

  # Initialize the argument parser
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-o', '--output', help='output file', dest='output', type=str, required=True)

  # Parse arguments
  args = argparser.parse_args()

  #pubmed = "https://www.ncbi.nlm.nih.gov/"


  name = 'output'
  #url = 'https://web.expasy.org/abcd/ABCD_AA001'
  #links = find_all_url(url)

  # Scrape each page from the database 
  counter = 0
  result = []
  for id in url_generator():

    # Generate the final url
    url = 'https://web.expasy.org/abcd/{}'.format(id)
    print('Fetching id {}'.format(id))

    # Scrape page
    data = scrape_url(url)
    if data is not None:
      # Append id & store
      data['id'] = id
      #data['pubmed'] = list_url[0]
      result.append(data)

    # Increment counter
    counter += 1
    if counter == 10:
      break

  # Store results
  #print(result[1])
  result = pandas.DataFrame.from_dict(result)
  #result.to_csv(args.output, sep='\t', index=False)
  result.to_csv(name, sep=';', index=False)

  # Done
  print('Scraped {} pages. Number of extracted entries: {}'.format(counter, len(result)))