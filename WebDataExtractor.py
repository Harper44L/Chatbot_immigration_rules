# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup

rules = []
def get_inf_from_web1(url1):
    res = requests.get(url1)
    ret = res.text
    soup = BeautifulSoup(ret, 'lxml')

    # extracted= (soup.select('.legislative-list'))
    extracted = get_label_rules(soup, 'legislative-list')
    # print(extracted)

    for i in enumerate(extracted):
        # print(i)
        segment = i[1].text.strip()
        # print(segment)
        rules.append(segment)

    f = open('immigration.txt', 'w')
    for i in enumerate(rules):
        f.write(i[1])
        f.write("\n")
    f.close()


# extracted = (soup.select('.legislative-list'))
def get_label_rules(soup, label):
    rules = []
    s = "." + label
    return (soup.select(s))


def get_inf_from_web2(url2):
    res = requests.get(url2)
    ret = res.text
    soup = BeautifulSoup(ret, 'lxml')
    # extracted= (soup.select('.legislative-list'))
    extracted1 = get_label_rules(soup, 'gem-c-govspeak')
    extracted2 = get_label_rules(soup, "gem-c-step-nav__step")

    # 要用的：
    for i in enumerate(extracted2):
        segment=i[1].text.strip()
        rules.append(segment)
    for i in enumerate(extracted1):
        segment = i[1].text.strip()
        print(segment)
        rules.append(segment)

appendix = 'https://www.gov.uk/student-visa/course'
get_inf_from_web2(appendix)

f = open('immigration.txt', 'w')
for i in enumerate(rules):
    if(len(i[1])>1):
        f.write(i[1])
        f.write("\n")
f.close()

# #
# visa = "https://www.gov.uk/student-visa"
# get_inf_from_web2(visa)
#
# course="https://www.gov.uk/student-visa/course"
# get_inf_from_web2(course)
#
# money="https://www.gov.uk/student-visa/money"
# get_inf_from_web2(money)
#
# english="https://www.gov.uk/student-visa/knowledge-of-english"
# get_inf_from_web2(english)
#
# documents="https://www.gov.uk/student-visa/documents-you-must-provide"
# get_inf_from_web2(documents)
#
# apply="https://www.gov.uk/student-visa/apply"
# get_inf_from_web2(apply)
#
# members="https://www.gov.uk/student-visa/family-members"
# get_inf_from_web2(members)
#
# extend="https://www.gov.uk/student-visa/extend-your-visa"
# get_inf_from_web2(extend)
#
# switch="https://www.gov.uk/student-visa/switch-to-this-visa"
# get_inf_from_web2(switch)





