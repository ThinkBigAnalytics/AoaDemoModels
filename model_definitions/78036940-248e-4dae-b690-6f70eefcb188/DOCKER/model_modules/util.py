
def execute_sql_script(conn, filename):
    with open(filename, 'r') as f:
        cmds = f.read().split(';')
        
    for cmd in cmds:
        cmd = cmd.strip
        if cmd:
            print("Executing cmd: {}".format(cmd))
            conn.execute(cmd)

    