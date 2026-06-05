const { DynamoDBClient } = require("@aws-sdk/client-dynamodb");
const { DynamoDBDocumentClient, DeleteCommand } = require("@aws-sdk/lib-dynamodb");

const client = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(client);

const TABLE_NAME = process.env.DYNAMODB_TABLE || "FaceAuthUsers";

exports.handler = async (event) => {
    console.log("Delete User Sync Triggered:", JSON.stringify(event));

    try {
        const body = typeof event.body === "string" ? JSON.parse(event.body) : event.body;
        const { user_id } = body;

        if (!user_id) {
            return {
                statusCode: 400,
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ error: "Missing user_id in delete payload." })
            };
        }

        // Delete user profile and encrypted embedding in DynamoDB
        await docClient.send(new DeleteCommand({
            TableName: TABLE_NAME,
            Key: { user_id }
        }));

        return {
            statusCode: 200,
            headers: {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            body: JSON.stringify({ message: "User deleted successfully in cloud registry.", user_id })
        };
    } catch (error) {
        console.error("Delete failed:", error);
        return {
            statusCode: 500,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ error: "Internal Server Error: " + error.message })
        };
    }
};
