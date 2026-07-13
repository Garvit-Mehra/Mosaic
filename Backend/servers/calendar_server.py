#!/usr/bin/env python3
import os
import logging
from mcp.server.fastmcp import FastMCP
from fastapi import HTTPException
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import datetime

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Calendar API setup
SCOPES = ['https://www.googleapis.com/auth/calendar']
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), 'credentials.json')
TOKEN_FILE = os.path.join(os.path.dirname(__file__), 'token.json')
F1_CALENDAR_ID = "329f505a35c271c59fb3a403ea2a01c9b782e05104f62ceb8f81953c34d266ae@group.calendar.google.com"
IITH_CALENDAR_ID = "8a98968ba0d620ff2fecd9e7325cdd83369bbeb6d67c663063bdc01f33feeb56@group.calendar.google.com"


def get_calendar_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    return build('calendar', 'v3', credentials=creds)

# MCP server
mcp = FastMCP(
    name="Mosaic Calendar Server",
    host="127.0.0.1",
    port=8010
)

# =============================
# Pydantic Models for Calendar
# =============================
class EventTime(BaseModel):
    dateTime: str  # ISO format
    timeZone: Optional[str] = "Asia/Kolkata"

class RecurrenceRule(BaseModel):
    freq: str  # e.g., 'DAILY', 'WEEKLY', 'MONTHLY'
    interval: Optional[int] = 1
    count: Optional[int] = None
    until: Optional[str] = None  # ISO date
    byweekday: Optional[List[str]] = None

    def to_rrule(self):
        rule = f"RRULE:FREQ={self.freq.upper()}"
        if self.interval:
            rule += f";INTERVAL={self.interval}"
        if self.count:
            rule += f";COUNT={self.count}"
        if self.until:
            rule += f";UNTIL={self.until.replace('-', '')}"
        if self.byweekday:
            rule += f";BYDAY={','.join(self.byweekday)}"
        return rule

class CreateEventRequest(BaseModel):
    summary: str
    description: Optional[str] = None
    start: EventTime
    end: EventTime
    location: Optional[str] = None
    attendees: Optional[List[str]] = None
    color: Optional[str] = None  # Google Calendar colorId
    recurrence: Optional[RecurrenceRule] = None
    calendar_hint: Optional[str] = None  # e.g., 'F1', 'IITH', or leave blank for auto

class EditEventRequest(BaseModel):
    event_id: str
    calendar_hint: Optional[str] = None
    updates: Dict[str, Any]

class ViewEventsRequest(BaseModel):
    calendar_hint: Optional[str] = None
    time_min: Optional[str] = None  # ISO format
    time_max: Optional[str] = None  # ISO format
    max_results: Optional[int] = 10

class DeleteEventRequest(BaseModel):
    event_id: str
    calendar_hint: Optional[str] = None

class FindEventsRequest(BaseModel):
    query: str
    calendar_hint: Optional[str] = None
    time_min: Optional[str] = None
    time_max: Optional[str] = None
    max_results: Optional[int] = 10

class GetEventDetailsRequest(BaseModel):
    event_id: str
    calendar_hint: Optional[str] = None

class FreeBusyRequest(BaseModel):
    calendar_hint: Optional[str] = None
    time_min: str  # ISO format
    time_max: str  # ISO format

# =============================
# Calendar Utility Functions
# =============================
def choose_calendar_id(summary: str, calendar_hint: Optional[str] = None) -> str:
    """Decide which calendar to use based on event summary or hint."""
    if calendar_hint:
        if calendar_hint.lower() == 'f1':
            return F1_CALENDAR_ID
        if calendar_hint.lower() == 'iith':
            return IITH_CALENDAR_ID
    # Heuristic: if 'f1' in summary, use F1 calendar, else IITH
    if 'f1' in summary.lower():
        return F1_CALENDAR_ID
    return IITH_CALENDAR_ID

# =============================
# MCP Tools for Calendar Events
# =============================
@mcp.tool(
    name="view_events",
    description="View upcoming events from the appropriate Google Calendar. Optionally filter by time range or calendar type."
)
async def view_events(request: ViewEventsRequest) -> Dict[str, Any]:
    try:
        service = get_calendar_service()
        calendar_id = choose_calendar_id("", request.calendar_hint)
        now = request.time_min or datetime.datetime.utcnow().isoformat() + 'Z'
        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=now,
            timeMax=request.time_max,
            maxResults=request.max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])
        return {"events": events}
    except Exception as e:
        logger.error(f"Error viewing events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mcp.tool(
    name="create_event",
    description="Create a new event in the appropriate Google Calendar. Supports color and recurrence."
)
async def create_event(request: CreateEventRequest) -> Dict[str, Any]:
    try:
        service = get_calendar_service()
        calendar_id = choose_calendar_id(request.summary, request.calendar_hint)
        event = {
            'summary': request.summary,
            'description': request.description,
            'start': request.start.dict(),
            'end': request.end.dict(),
        }
        if request.location:
            event['location'] = request.location
        if request.attendees:
            event['attendees'] = [{'email': email} for email in request.attendees]
        if request.color:
            event['colorId'] = request.color
        if request.recurrence:
            event['recurrence'] = [request.recurrence.to_rrule()]
        created = service.events().insert(calendarId=calendar_id, body=event).execute()
        return {"event": created}
    except Exception as e:
        logger.error(f"Error creating event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mcp.tool(
    name="edit_event",
    description="Edit an existing event in the appropriate Google Calendar. Specify event_id and updates."
)
async def edit_event(request: EditEventRequest) -> Dict[str, Any]:
    try:
        service = get_calendar_service()
        # For edit, must know which calendar. Try to guess if not provided.
        calendar_id = choose_calendar_id("", request.calendar_hint)
        event = service.events().get(calendarId=calendar_id, eventId=request.event_id).execute()
        for k, v in request.updates.items():
            if k in ["start", "end"] and isinstance(v, dict):
                event[k] = v
            else:
                event[k] = v
        updated = service.events().update(calendarId=calendar_id, eventId=request.event_id, body=event).execute()
        return {"event": updated}
    except Exception as e:
        logger.error(f"Error editing event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mcp.tool(
    name="delete_event",
    description="Delete an event from the appropriate Google Calendar by event_id."
)
async def delete_event(request: DeleteEventRequest) -> Dict[str, Any]:
    try:
        service = get_calendar_service()
        calendar_id = choose_calendar_id("", request.calendar_hint)
        service.events().delete(calendarId=calendar_id, eventId=request.event_id).execute()
        return {"message": f"Event {request.event_id} deleted from calendar."}
    except Exception as e:
        logger.error(f"Error deleting event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mcp.tool(
    name="find_events_by_query",
    description="Find events by keyword in summary or description, with optional time range and calendar selection."
)
async def find_events_by_query(request: FindEventsRequest) -> Dict[str, Any]:
    try:
        service = get_calendar_service()
        calendar_id = choose_calendar_id(request.query, request.calendar_hint)
        now = request.time_min or datetime.datetime.utcnow().isoformat() + 'Z'
        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=now,
            timeMax=request.time_max,
            maxResults=request.max_results,
            singleEvents=True,
            orderBy='startTime',
            q=request.query
        ).execute()
        events = events_result.get('items', [])
        return {"events": events}
    except Exception as e:
        logger.error(f"Error finding events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mcp.tool(
    name="get_event_details",
    description="Get full details for a specific event by event_id."
)
async def get_event_details(request: GetEventDetailsRequest) -> Dict[str, Any]:
    try:
        service = get_calendar_service()
        calendar_id = choose_calendar_id("", request.calendar_hint)
        event = service.events().get(calendarId=calendar_id, eventId=request.event_id).execute()
        return {"event": event}
    except Exception as e:
        logger.error(f"Error getting event details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mcp.tool(
    name="get_free_busy_times",
    description="Get free/busy times for the specified calendar and time range."
)
async def get_free_busy_times(request: FreeBusyRequest) -> Dict[str, Any]:
    try:
        service = get_calendar_service()
        calendar_id = choose_calendar_id("", request.calendar_hint)
        body = {
            "timeMin": request.time_min,
            "timeMax": request.time_max,
            "items": [{"id": calendar_id}]
        }
        result = service.freebusy().query(body=body).execute()
        return {"calendars": result.get("calendars", {})}
    except Exception as e:
        logger.error(f"Error getting free/busy times: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":    
    # Start the MCP server
    logger.info("Starting Mosaic Calendar Server...")
    mcp.run(transport="sse")
