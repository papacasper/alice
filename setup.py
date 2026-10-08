import random, os, json, sys
R=sys.argv[1]; random.seed(11)
for d in ("logs","conf","data","docs"): os.makedirs(f"{R}/{d}",exist_ok=True)
errs={"auth":14,"billing":37,"search":29,"mail":22,"cdn":35}
lines=[];hi=0
for s,n in errs.items():
    for _ in range(n):
        c=random.randint(100,599)
        if s=="billing" and c>=500: hi+=1
        lines.append(f"2026-10-0{random.randint(1,6)} {random.randint(0,23):02d}:{random.randint(0,59):02d} ERROR [{s}] failure code {c}")
    for _ in range(90): lines.append(f"2026-10-0{random.randint(1,6)} {random.randint(0,23):02d}:{random.randint(0,59):02d} INFO [{s}] request ok latency={random.randint(5,900)}ms")
random.shuffle(lines)
open(f"{R}/logs/app.log","w").write("\n".join(lines)+"\n")
open(f"{R}/conf/services.toml","w").write("".join(f"[{s}]\nport = {8100+i*37}\nowner = \"team-{s}\"\n" for i,s in enumerate(errs)))
rows=[(f"2026-09-{d:02d}",round(random.uniform(10,500),2)) for d in range(1,31)]
open(f"{R}/data/sales.csv","w").write("date,amount\n"+"\n".join(f"{d},{a}" for d,a in rows)+"\n")
open(f"{R}/docs/README.md","w").write("Internal services overview. See conf/services.toml for ports.\n")
json.dump({"top":"billing","port":8100+1*37,"hi":hi,"sales":round(sum(a for _,a in rows),2),"ports":[8100+i*37 for i in range(5)]},open(f"{R}/truth.json","w"))
print(open(f"{R}/truth.json").read())
