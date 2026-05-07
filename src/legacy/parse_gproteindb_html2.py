from bs4 import BeautifulSoup
import re

html = open('E:/kimi/Kimi_Agent_批判者监督执行/gproteindb_coupling_datasets.html', 'r', encoding='utf-8').read()
soup = BeautifulSoup(html, 'html.parser')

with open('E:/kimi/Kimi_Agent_批判者监督执行/gproteindb_parse_debug.txt', 'w', encoding='utf-8') as out:
    table = soup.find('table', {'id': 'data_overview'})
    if table:
        tbody = table.find('tbody')
        for i, tr in enumerate(tbody.find_all('tr')[:3]):
            out.write(f'--- row {i} html ---\n')
            out.write(tr.prettify()[:2000])
            out.write('\n\n')

    scripts = soup.find_all('script')
    for s in scripts:
        txt = s.string or ''
        if 'data_overview' in txt or 'Datatable' in txt or 'signprot' in txt:
            if len(txt) < 5000:
                out.write('--- script snippet ---\n')
                out.write(txt)
                out.write('\n\n')
            else:
                lines = [l for l in txt.splitlines() if 'url' in l.lower() or 'data_overview' in l or 'ajax' in l.lower()]
                if lines:
                    out.write('--- script lines ---\n')
                    for l in lines:
                        out.write(l + '\n')
                    out.write('\n')

print('Debug output saved to gproteindb_parse_debug.txt')
