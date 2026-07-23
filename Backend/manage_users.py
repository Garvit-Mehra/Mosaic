#!/usr/bin/env python3
"""
Mosaic User Management Script

Usage:
    python manage_users.py list
    python manage_users.py add <username> <email> <password> [--role admin|user] [--verified]
    python manage_users.py delete <username>
    python manage_users.py verify <username>
    python manage_users.py set-role <username> <admin|user>
    python manage_users.py reset-password <username> <new_password>
    python manage_users.py info <username>
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

from utils.UserDB import UserManager


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    db = UserManager()
    command = sys.argv[1]

    if command == "list":
        users = db.list_users()
        if not users:
            print("No users found.")
            return
        print(f"{'ID':<4} {'Username':<20} {'Email':<30} {'Role':<8} {'Verified':<10} {'Created'}")
        print("-" * 100)
        for u in users:
            print(f"{u['id']:<4} {u['username']:<20} {u['email']:<30} {u['role']:<8} {'✓' if u['verified'] else '✗':<10} {u['created_at']}")

    elif command == "add":
        if len(sys.argv) < 5:
            print("Usage: python manage_users.py add <username> <email> <password> [--role admin|user] [--verified]")
            sys.exit(1)

        username = sys.argv[2]
        email = sys.argv[3]
        password = sys.argv[4]
        role = "user"
        verified = False

        if "--role" in sys.argv:
            idx = sys.argv.index("--role")
            role = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "user"
        if "--verified" in sys.argv:
            verified = True

        try:
            user = db.create_user(username, email, password, role=role, verified=verified)
            print(f"✓ Created user: {user['username']} ({user['email']}) role={user['role']} verified={user['verified']}")
        except ValueError as e:
            print(f"✗ Error: {e}")
            sys.exit(1)

    elif command == "delete":
        if len(sys.argv) < 3:
            print("Usage: python manage_users.py delete <username>")
            sys.exit(1)

        username = sys.argv[2]
        confirm = input(f"Delete user '{username}'? (y/N): ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return

        if db.delete_user(username):
            print(f"✓ Deleted user: {username}")
        else:
            print(f"✗ User '{username}' not found.")
            sys.exit(1)

    elif command == "verify":
        if len(sys.argv) < 3:
            print("Usage: python manage_users.py verify <username>")
            sys.exit(1)

        username = sys.argv[2]
        if db.verify_user(username):
            print(f"✓ User '{username}' is now verified.")
        else:
            print(f"✗ User '{username}' not found.")
            sys.exit(1)

    elif command == "set-role":
        if len(sys.argv) < 4:
            print("Usage: python manage_users.py set-role <username> <admin|user>")
            sys.exit(1)

        username = sys.argv[2]
        role = sys.argv[3]
        if role not in ("admin", "user"):
            print("Role must be 'admin' or 'user'.")
            sys.exit(1)

        if db.update_user(username, role=role):
            print(f"✓ User '{username}' is now role={role}.")
        else:
            print(f"✗ User '{username}' not found.")
            sys.exit(1)

    elif command == "reset-password":
        if len(sys.argv) < 4:
            print("Usage: python manage_users.py reset-password <username> <new_password>")
            sys.exit(1)

        username = sys.argv[2]
        password = sys.argv[3]
        if db.update_user(username, password=password):
            print(f"✓ Password reset for '{username}'.")
        else:
            print(f"✗ User '{username}' not found.")
            sys.exit(1)

    elif command == "info":
        if len(sys.argv) < 3:
            print("Usage: python manage_users.py info <username>")
            sys.exit(1)

        username = sys.argv[2]
        user = db.get_user_by_username(username)
        if not user:
            print(f"✗ User '{username}' not found.")
            sys.exit(1)

        print(f"Username:  {user['username']}")
        print(f"Email:     {user['email']}")
        print(f"Role:      {user['role']}")
        print(f"Verified:  {'✓' if user['verified'] else '✗'}")
        print(f"Created:   {user['created_at']}")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
