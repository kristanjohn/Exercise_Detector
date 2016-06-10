import psycopg2
import urlparse
import StringIO
import json

# DATABASE FORMAT
# -----------------
# CREATE TABLE raw_data (
#     eid VARCHAR(50),
#     sid VARCHAR(50),
#     t VARCHAR(50),
#     ax VARCHAR(50),
#     ay VARCHAR(50),
#     az VARCHAR(50),
#     gx VARCHAR(50),
#     gy VARCHAR(50),
#     gz VARCHAR(50)
# );

def split_entries(entries):
    records = []
    record = []
    flag = 0

    for entry in entries:
        # start of record
        if (all(i == '1' for i in entry[1:])):
            flag = 1

        # end of record
        if (all(i == '0' for i in entry[1:])):
            records.append(record)
            record = []
            flag = 0

        # regular entry
        if (flag == 1):
            record.append(entry)

    return records

def clean_samples(record):
    ones = 0
    twos = 0
    target = 0
    extra = 0
    flag = 0
    data = []

    for entry in record:
        if (all(i == '0' for i in entry[1:])):
            flag = 1

        if (all(i == '1' for i in entry[1:])):
            pass

        elif (flag != 1):
            data.append(entry)

    for entry in data:
        if (entry[1] == '1'):
            ones += 1
        else:
            twos += 1

    if (ones == twos):
        return data
    elif (ones > twos):
        target = 1
        extra = ones - twos
    else:
        target = 2
        extra = twos - ones

    for entry in reversed(data):
        if (entry[1] == str(target)):
            data.remove(entry)
            extra -= 1
            if (extra == 0):
                return data


def create_json(record):
    #print record
    print "Parsing eid:", record[0][0]
    cleaned = clean_samples(record)
    # cleaned = record
    data = []
    e = {}
    for entry in cleaned:
        e['eid'] = entry[0]
        e['sid'] = entry[1]
        e['t'] = entry[2]
        e['ax'] = entry[3]
        e['ay'] = entry[4]
        e['az'] = entry[5]
        e['gx'] = entry[6]
        e['gy'] = entry[7]
        e['gz'] = entry[8]
        data.append(e)
        e = {}

    # remove last 4 items if data does not contain equal amount of sensor readings
    # if (data[0]['sid'] == data[-1]['sid']):
    #     data = data[:4-1]

    if (len(data) < 2):
        print("Malformed data.")
        return

    mapping = {}
    mapping[str(cleaned[0][0])] = "UNCATEGORISED"

    with open("categorise.csv") as f:
        while True:
            category = f.readline().strip()
            datasets = f.readline().strip()
            eids = datasets.strip().split(",")
            for eid in eids:
                mapping[eid.strip()] = category
            if not datasets: break #EOF

    mapping[''] = "UNCATEGORISED"

    with open("data/data_" + str(cleaned[0][0]) + "_" + mapping[str(cleaned[0][0])] + ".json", "w") as f:
        print "Writing", "data_" + str(cleaned[0][0])+ "_" + mapping[str(cleaned[0][0])] + ".json..."
        f.write("[")
        for item in data[:-1]:
            f.write("%s,\n" % json.dumps(item))
        f.write("%s]\n" % json.dumps(data[-1]))

######## PROGRAM START ##########

urlparse.uses_netloc.append("postgres")
url = urlparse.urlparse("postgres://obkenkootsqtvz:MaeQz3GQPJ1hIPknnNHcrgCNMB@ec2-54-83-27-147.compute-1.amazonaws.com:5432/d3nbtq2btt810j")

conn = psycopg2.connect(
    database=url.path[1:],
    user=url.username,
    password=url.password,
    host=url.hostname,
    port=url.port
)

cur = conn.cursor()

cur.execute("SELECT DISTINCT eid from raw_data;")

eids = []
for val in cur:
    eids.append(val[0])

for eid in eids:
    cur.execute("SELECT * FROM raw_data WHERE eid='%s';" % eid)
    create_json([entry for entry in cur])

conn.commit()
conn.close()

print "Done."
