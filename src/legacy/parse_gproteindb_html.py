from bs4 import BeautifulSoup
import re

html = open('E:/kimi/Kimi_Agent_批判者监督执行/gproteindb_coupling_datasets.html', 'r', encoding='utf-8').read()
soup = BeautifulSoup(html, 'html.parser')

table = soup.find('table', {'id': 'data_overview'})
if table:
    rows = table.find_all('tr')
    print(f'Total rows (including header): {len(rows)}')
    tbody = table.find('tbody')
    if tbody:
        trs = tbody.find_all('tr')
        print(f'Tbody rows: {len(trs)}')
        for tr in trs[:10]:
            tds = [td.get_text(strip=True) for td in tr.find_all('td')]
            print(' | '.join(tds[:5]))
            links = [a['href'] for a in tr.find_all('a', href=True)]
            if links:
                print('  Links:', links)
    else:
        print('No tbody found')
else:
    print('data_overview table not found')

scripts = soup.find_all('script')
for s in scripts:
    txt = s.string or ''
    if 'ajax' in txt.lower() or 'url' in txt.lower() or 'datatable' in txt.lower():
        urls = re.findall(r'url\s*:\s*["\']([^"\']+)["\']', txt)
        if urls:
            print('Ajax URLs found:', urls)
