def init_data():
    locations = {
        "entrance 1": "near the north road, main entrance",
        "entrance 2": "near the south-east side, by admin block",
        "flag post": "middle of Entrance 2 and Admin Block",
        "admin block": "central administrative offices",
        "parents stay area": "near Junction for guests",
        "engineering admin block": "administrative area for engineering faculty",
        "faculty quarters": "housing for faculty members",
        "hostel 1": "student accommodation near sports ground",
        "food court": "canteen and fitness area close to Hostel 1",
        "food court & gym": "canteen and fitness area close to Hostel 1",
        "sports area and ground": "sports complex near Hostel and Gym"
    }
    return locations

def get_location_response(user_input, locations):
    for name, desc in locations.items():
        if name in user_input:
            return f"{name.title()} is {desc}."
    return "Location not found."

def get_faq():
    return "You can ask me about campus locations. For example, 'Where is Hostel 1?' or 'How to go to Admin Block?'"

def main():
    locations = init_data()
    print("Welcome to Chanakya University Chatbot!")
    print("Type 'exit' to quit.")
    while True:
        user_input = input("You: ").lower()
        if user_input == 'exit':
            print("Bot: Goodbye!")
            break
        elif "where is" in user_input or "how to go to" in user_input:
            response = get_location_response(user_input, locations)
        elif "help" in user_input or "faq" in user_input:
            response = get_faq()
        else:
            response = "Sorry, I didn't understand that. Please ask about locations."
        print("Bot:", response)

if __name__ == "__main__":
    main()
