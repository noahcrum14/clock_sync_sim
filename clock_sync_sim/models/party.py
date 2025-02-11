import numpy as np

class Party:
    def __init__(self, worldtime, party_id, source_type, rep_rate, delay, offset):
        self.worldtime = worldtime
        self.party_id = party_id
        self.source = source_type
        self.rep_rate = rep_rate  # In Hz (10 MHz = 10^7 Hz)
        self.event_spacing_ns = int(1e9 / self.rep_rate)  # Convert Hz to ns
        self.delay = delay  # In ns
        self.offset = offset  # In ns
        self.interval_length = 30000
        self.pols = ['h', 'v', 'd', 'a']
        self.velocity = 0.20818920694444445  # c in fiber [m/ns]

    def pol_selection(self):
        """Select a random polarization state."""
        return np.random.choice(self.pols)

    def emission_schedule(self):
        """Generate an emission schedule with timestamps, polarization, and photon count."""
        emit_schedule = [
            {'t': i * self.event_spacing_ns + self.delay, 'pol': self.pol_selection(), 'photons': 1}
            for i in range(self.interval_length)
        ]
        return emit_schedule

    def clock_reading(self, schedule, index):
        """Determine the local time reading of a schedule event."""
        event = schedule[index]['t']
        return event + self.offset

    def propagate(self, schedule, distance):
        """Add the propagation time to the schedule."""
        for i in range(len(schedule)):
            schedule[i]['t'] += distance / self.velocity
        return schedule

    def give_pol_record(self, schedule):
        """Return a list of polarization states from a schedule."""
        return [schedule[i]['pol'] for i in range(len(schedule.keys()))]