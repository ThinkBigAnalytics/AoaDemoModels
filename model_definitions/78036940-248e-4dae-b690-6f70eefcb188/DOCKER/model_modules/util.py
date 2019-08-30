from jinja2 import Template

def template_sql_script(filename, jinja_ctx):
    with open(filename) as f:
        template = Template(f.read())
    
    return template.render(jinja_ctx)


def execute_sql_script(conn, filename, jinja_ctx):
    script = template_sql_script(filename, jinja_ctx)
    
    stms = script.split(';')
        
    for stm in stms:
        stm = stm.strip()
        if stm:
            print("Executing statement: {}".format(stm))
            
            try:
                conn.execute(stm)
            except Exception as e:
                if stm.startswith("DROP"):
                    print("Ignoring DROP statement exception")
                else:
                    raise e

    