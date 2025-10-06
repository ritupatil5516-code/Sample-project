def verify(plan, results):
    missing=[k for k,v in results.items() if isinstance(v,dict) and v.get('error')]; return (len(missing)==0, missing)
