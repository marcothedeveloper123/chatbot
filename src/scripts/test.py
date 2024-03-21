import requests
from bs4 import BeautifulSoup

def get_fifa_worldcup_details():
	url = "https://en.wikipedia.org/wiki/FIFA_World_Cup"

	r = requests.get(url)

	if not r.ok:  # check the status of response
		return "Unable to fetch the page."

	soup = BeautifulSoup(r.text, 'html.parser')

	table = soup.find('table', attrs={'class': ['wikitable']}) # updated css selector

	if not table:
		return "Unable to find FIFA World Cup details"

	next_worldcups = []

	rows = table.find_all(['tr'])

	for row in rows[1:-2]:  # skip header and last two lines
		cells = row.find_all('td') # updated css selector

		if len(cells) != 3:
			continue

		next_worldcups.append({
				'Year': cells[0].text.strip(),
				'Country': cells[1].get_text(separator=" ", strip=True),
				'Date': cells[2].get_text(separator=" ", strip=True)
			})

	return next_worldcups

# Testing the function
print(get_fifa_worldcup_details())
