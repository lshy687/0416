import requests
url = "https://its4.pku.edu.cn/cas/ITSClient"
payload = {
    # 填写账号和密码
    'username': '2201111610',
    'password': 'li19970112',
    'iprange': 'free',
    'cmd': 'open'
}
headers = {'Content-type': 'application/x-www-form-urlencoded'}
result = requests.post(url, params=payload, headers=headers)
print(result.text)