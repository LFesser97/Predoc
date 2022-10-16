import argparse

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='command')
login = subparser.add_parser('login')
register = subparser.add_parser('register')

login.add_argument('--username', type=str, required=True)
login.add_argument('--password', type=str, required=True)

register.add_argument('--firstname', type=str, required=True)
register.add_argument('--lastname', type=str, required=True)
register.add_argument('--username', type=str, required=True)
register.add_argument('--email', type=str, required=True)
register.add_argument('--password', type=str, required=True)

args = parser.parse_args()

if args.command == 'login':
  print('Logging in with username:', args.username,
  'and password:', args.password)
  
elif args.command == 'register':
  print('Creating username', args.username,
  'for new member', args.firstname, args.lastname,
  'with email:', args.email,
  'and password:', args.password)