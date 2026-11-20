"""Local administrator bootstrap. Run from your private server terminal."""
import argparse
import getpass
import os
from pathlib import Path
from auth_security import UserAuth
from store import Store


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--db',default=os.getenv('BUILDER_DB',str(Path(__file__).resolve().parent/'data'/'builder.sqlite3')))
    sub=p.add_subparsers(dest='action',required=True)
    for cmd in ('bootstrap-admin','add-user'):
        s=sub.add_parser(cmd);s.add_argument('username')
    for cmd in ('grant','revoke'):
        s=sub.add_parser(cmd);s.add_argument('project_id');s.add_argument('username')
        if cmd=='grant':s.add_argument('role',choices=['viewer','editor','owner'])
    args=p.parse_args(argv)
    auth=UserAuth(Store(args.db))
    if args.action in ('bootstrap-admin','add-user'):
        password=getpass.getpass('New password (14+ characters): ')
        confirmation=getpass.getpass('Confirm password: ')
        if password!=confirmation:raise SystemExit('Passwords do not match.')
        data=auth.add_user(args.username,password,admin=args.action=='bootstrap-admin')
        print('Created',data['username'],'admin=',data['admin'])
    elif args.action=='grant':
        auth.grant(args.project_id,args.username,args.role)
        print('Grant recorded')
    else:
        auth.revoke(args.project_id,args.username)
        print('Access revoked')

if __name__=='__main__':main()
