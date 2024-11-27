# use dictionary to store customer name and balance
money_dict = {}
# use dictionary to store customer name and transaction history
transaction_dict = {}

# add customer name and initial balance
def add_customer():
    while True:
        name = input("Enter customer name : ")
        money = int(input(f"Enter initial balance for {name} : "))
        money_dict[name] = money
        transaction_dict[name] = []
        add_transaction(name, money, "deposit")
        response = input("Do you want to add another customer? (y/n)")
        if response == 'n':
            break

# print customer name and balance
def print_customer():
    print("Current balance of customers : ")
    for customer, balance in money_dict.items():
        print(f"{customer} : {balance}")

# deposit money to customer's account
def deposit():
    name = input("Enter customer name : ")
    amount = int(input("Enter deposit amount : "))
    money_dict[name] += amount
    add_transaction(name, amount, "deposit") 
    print(f"{name}'s new balance is {money_dict[name]}")

# withdraw money from customer's account
def withdraw():
    name = input("Enter customer name : ")
    amount = int(input("Enter the amount you want to withdraw : "))
    if money_dict[name] < amount:
        print("Insufficient balance")
        return
    money_dict[name] -= amount
    add_transaction(name, amount, "withdrawal")
    print(f"{name}'s new balance is {money_dict[name]}")

# record transaction with date
def add_transaction(name, amount, transaction_type):
    date = input("Enter date in YYYYMMDD format : ")
    transaction_dict[name].append((date, amount, transaction_type))

# view transaction history of specific customer by date
def view_transaction():
    name = input("Enter customer name : ")
    print(f"Transaction history of {name} : ")
    for date, amount, transaction_type in transaction_dict[name]:
        print(f"{date}: {transaction_type} {amount}")

# transfer money from one customer to another
def transfer():
    sender = input("Enter sender's name : ")
    receiver = input("Enter receiver's name : ")
    amount = int(input("Enter amount to transfer : "))
    if money_dict[sender] < amount:
        print("Insufficient balance")
        return
    money_dict[sender] -= amount
    money_dict[receiver] += amount
    add_transaction(sender, amount, "transfer out")
    add_transaction(receiver, amount, "transfer in")
    print(f"{sender}'s new balance : {money_dict[sender]}")
    print(f"{receiver}'s new balance : {money_dict[receiver]}")

# main
menu = """
Menu : 
        1. Add customers
        2. List customers
        3. Deposit money
        4. Withdraw money
        5. View transactions by date
        6. Transfer money
        7. Exit
        """

while True:
    print(menu)
    command = int(input("Enter command(1~7) : "))
    if command == 1:
        add_customer()
    elif command == 2:
        print_customer()
    elif command == 3:
        deposit()
    elif command == 4:
        withdraw()
    elif command == 5:
        view_transaction()
    elif command == 6:
        transfer()
    elif command == 7:
        break
    else:
        print("Invalid command")

print("END")


