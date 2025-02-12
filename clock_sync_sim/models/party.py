import numpy as np
from ..utils.calculations import *
from ..config.settings import SIGMA, C_FIBER, SIGMA_JITTER

class Party:
    def __init__(self, worldtime, party_id, source_type, rep_rate, delay, offset, mu=1):
        self.worldtime = worldtime
        self.party_id = party_id
        self.source = source_type
        self.mu = mu
        self.rep_rate = rep_rate  # In Hz (10 MHz = 10^7 Hz)
        self.event_spacing_ns = int(1e9 / self.rep_rate)  # Convert Hz to ns
        self.delay = delay  # In ns
        self.offset = offset  # In ns
        self.interval_length = 5000
        self.pols = ['h', 'v', 'd', 'a']
        self.velocity = 0.20818920694444445  # c in fiber [m/ns]
        self.wavelength = 1550 * 10 ** -9 #wavelength in [m]
        self.d_lambda = self.wavelength**2/self.velocity * (SIGMA * 10 ** 12)

    def pol_selection(self):
        """Select a random polarization state."""
        return np.random.choice(self.pols)

    def get_photon_num(self):
        if self.source == 'coherent':
            return np.random.poisson(np.sqrt(self.mu))
        else:
            return 1

    def emission_schedule(self):
        """Generate an emission schedule with timestamps, polarization, and photon count."""
        emit_schedule = [
            {'t': i * self.event_spacing_ns + self.delay, 'pol': self.pol_selection(), 'photons': self.get_photon_num()}
            for i in range(self.interval_length)
        ]
        return emit_schedule

    def clock_reading(self, schedule, index):
        """Determine the local time reading of a schedule event."""
        event = schedule[index]['t']
        return event + self.offset + jitter(SIGMA_JITTER)

    def propagate(self, schedule, distance):
        """Add the propagation time to the schedule."""
        for i in range(len(schedule)):
            schedule[i]['t'] += distance / self.velocity \
                                + jitter(SIGMA_JITTER) \
                                + chromatic_dispersion(distance, self.d_lambda)\
                                + pmd(distance)
        return schedule

    def give_pol_record(self, schedule):
        """Return a list of polarization states from a schedule."""
        return [schedule[i]['pol'] for i in range(len(schedule))]