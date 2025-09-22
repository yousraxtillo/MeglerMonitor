import requests, pandas as pd

url = "https://dnbeiendom.no/api/v1/cognitivesearch/departments"
headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json;charset=UTF-8",
}
payload = {"filter": "hide eq false", "orderBy": ["name asc"], "top": 200}

resp = requests.post(url, headers=headers, json=payload, timeout=30)
resp.raise_for_status()
data = resp.json()

rows = []
for dep in data.get("documents", []):
    dep_id   = dep.get("id")
    dep_name = dep.get("name")
    dep_email = dep.get("email")
    dep_phone = dep.get("phone")
    dep_path  = (dep.get("cms") or {}).get("path")
    coords    = dep.get("coordinates", {}).get("coordinates", [None, None])
    lon, lat  = (coords + [None, None])[:2]

    # hent ut noen location-felter (gate/zip/by)
    street = zipc = city = None
    for loc in dep.get("locations", []):
        if loc.get("type") == "STREET":
            street = loc.get("value")
        elif loc.get("type") == "ZIPCODE":
            zipc = loc.get("value")
        elif loc.get("type") == "CITY":
            city = loc.get("value")

    # én rad per megler i avdelingen
    for br in dep.get("brokers", []):
        rows.append({
            "department_id": dep_id,
            "department_name": dep_name,
            "department_email": dep_email,
            "department_phone": dep_phone,
            "department_path": dep_path,
            "street": street,
            "zip": zipc,
            "city": city,
            "lat": lat,
            "lon": lon,
            "broker_id": br.get("id"),
            "broker_externalId": br.get("externalId"),
            "broker_name": br.get("name"),
            "broker_title": br.get("title"),
            "broker_mobile": br.get("mobilePhone"),
            "broker_email": br.get("email"),
            "broker_student": br.get("student"),
        })

df = pd.DataFrame(rows)
df.to_csv("dnbeiendom_brokers.csv", index=False)
print("Lagret:", len(df), "rader")
